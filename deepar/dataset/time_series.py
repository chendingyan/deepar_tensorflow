import time
import logging
import typing
import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Model

from deepar.dataset import Dataset
from deepar.dataset.utils import robust_timedelta, robust_reindex

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

pd.options.mode.chained_assignment = None  # default='warn'


class TimeSeriesTrain(Dataset):

    def __init__(
            self,
            pandas_df: pd.DataFrame,
            target_idx: int,
            timestamp_idx: int,
            grouping_idx: int = None,
            one_hot_indices: typing.List[int] = None,
            index_col: int = None,
            count_data: bool = False,
            negative_obs: int = 1,
            val_split: float = 0.2,
            freq: str = 'S',
            normalize_dates: bool = True
    ):
        """ prepares TimeSeriesTrain data object - augments data with covariates, standardizes data, encodes 
                categorical variables (for embeddings in model), optionally splits into training and validation sets
        
        Arguments:
            pandas_df {pd.DataFrame} -- df from which to sample time series
            target_idx {int} -- index of target column
            timestamp_idx {int} -- index of column containing timestamps
        
        Keyword Arguments:
            grouping_idx {int} -- index of grouping column (default: {None})
            one_hot_indices {typing.List[int]} -- list of indices of columns that are one hot encoded (default: {None})
            index_col {int} -- index column, if it exists, will be dropped (default: {None})
            count_data {bool} -- boolean indicating whether data is count data (determines loss function) (default: {False})
            negative_obs {int} -- how far before beginning of time series is it possible to set t for sampled series (default: {1})
            val_split {float} -- proportion of data to withhold for validation (default: {0.2})
            freq {str} -- frequency of the time series (default: {'S'})
            normalize_dates {bool} -- whether to normalize start/end dates to midnight before generating date ranges (default: {True})
        """

        super().__init__()

        if pandas_df is None:
            raise ValueError('Must provide a Pandas df to instantiate this class')

        # store constructor arguments as instance variables
        self._data = pandas_df
        self._one_hot_indices = one_hot_indices
        self._negative_obs = negative_obs
        col_names = list(self._data)
        if grouping_idx is None:
            self._data['category'] = 'dummy_cat'
            self._grouping_name = 'category'
            self._grouping_idx = len(col_names)
        else:
            self._grouping_name = col_names[grouping_idx]
            self._grouping_idx = grouping_idx
        self._data[self.grouping_name] = self._data[self._grouping_name].astype(str)
        self._timestamp_idx = timestamp_idx
        self._count_data = count_data
        self._freq = freq
        self._normalize_dates = normalize_dates

        # set name of target variable
        if target_idx is None:
            raise ValueError('Must provide an index for a target column to instantiate this class')
        self._data = self._data.rename(columns={col_names[target_idx]: 'target'})

        # set mask value
        if count_data:
            self._mask_value = 0
        else:
            self._mask_value = self._data['target'].min() - 1

        # delete index column if one exists (absolute position information only available through covariates)
        self._index_col = index_col
        if index_col is not None:
            self._data = self._data.drop(col_names[index_col], axis=1)

        # augment dataset with covariates
        self._time_name = self._sort_by_timestamp(col_names)
        # self._data = self._datetime_interpolation(self._data,
        #                                           self._data[self._time_name].iloc[-1],
        #                                           negative_offset=self._negative_obs
        #                                           )
        # 为什么需要一个dummy_test_category
        self._data = self._datetime_augmentation(self._data)
        self._age_augmentation()

        # need embeddings for cats in val, even if not in train
        # 1 extra for test cats not included in train or val
        self._unique_cats = self._data[self._grouping_name].unique()
        self._num_cats = len(self._data[self._grouping_name].unique()) + 1

        # convert groups to ints
        self._label_encoder = LabelEncoder()
        cat_names = self._data[self._grouping_name].astype(str).append(pd.Series(['dummy_test_category']))
        self._label_encoder.fit(cat_names)
        self._data[self._grouping_name] = self._label_encoder.transform(self._data[self._grouping_name])

        # self._label_encoder.fit_transform(self._data[self._grouping_name])
        # split into train + validation sets, create sampling dist.
        # 根据比例分一下train valid 但是是按照时间顺序排序区分的
        self._train_val_split(val_split)
        self._store_target_means(val_split)
        #
        # # standardize
        self._standardize(val_split)
        #
        # # store number of features and categorical count and target means
        self._features = self._data.shape[1] - 2  # -3 :target, grouping name, datetime, +1 :prev target
        self._count_data = count_data

    def _sort_by_timestamp(
            self,
            col_names: typing.List[str]
    ) -> str:

        """
        util function
            sort df by timestamp
        """

        # get name of time column
        if self._timestamp_idx is None:
            raise ValueError('Must provide the index of the timestamp column to instantiate this class')
        time_name = col_names[self._timestamp_idx]

        # sort data by time column
        self._data = self._data.sort_values(by=time_name)

        # convert to pd datetime objects
        if isinstance(self._data[time_name].iloc[0], int):
            self._data[time_name] = pd.to_datetime(self._data[time_name] - 1, unit='D')
            self._freq = 'D'
        elif isinstance(self._data[time_name].iloc[0], float):
            self._data[time_name] = pd.to_datetime(self._data[time_name], unit='s')

        return time_name

    def _datetime_interpolation(
            self,
            df: pd.DataFrame,
            max_date: datetime,
            min_date: datetime = None,
            negative_offset: int = 0
    ) -> pd.DataFrame:

        """
        util function
            interpolate along time dimension (to max_date) to create evenly spaced observations.
            this will interpolate covariates
        """

        # make sure negative offset is positive
        negative_offset = max(0, negative_offset)

        # interpolate all series to max_date 
        new_dfs = []
        for group, df in df.groupby(self._grouping_name):

            # find minimum date from this series
            if min_date is None:
                min_date_group = df[self._time_name].iloc[0] - robust_timedelta(negative_offset, self._freq)
            else:
                min_date_group = min_date

            # average duplicate timestamps before reindexing
            if sum(df[self._time_name].duplicated()) > 0:
                df = df.groupby(self._time_name).mean()
                df.insert(self._grouping_idx - 1, self._grouping_name, group)
            else:
                df.index = df[self._time_name]
                df = df.drop(self._time_name, axis=1)

            # reindex
            df_new = robust_reindex(df,
                                    min_date_group,
                                    max_date,
                                    freq=self._freq,
                                    normalize_dates=self._normalize_dates
                                    )
            df_new.insert(self._timestamp_idx, self._time_name, df_new.index.get_level_values(-1))
            df_new = df_new.reset_index(drop=True)

            # interpolate non-target columns
            if df.shape[0] > 1:
                try:
                    df_new.loc[:, df_new.columns != 'target'] = df_new.loc[:, df_new.columns != 'target'].interpolate()
                except ValueError:
                    logger.debug('There are no NA target values for which linear interpolation makes sense')
                df_new.loc[:, df_new.columns != 'target'] = df_new.loc[:, df_new.columns != 'target'].ffill().bfill()
            else:
                # edge case when new reindexed df only has one row
                replace = [col for col in df_new.columns if col != self._time_name]
                df_new.loc[:, replace] = df[replace].values[0]

            new_dfs.append(df_new)
        return pd.concat(new_dfs)

    def _age_augmentation(self):
        """
        util function
        增加该时间序列这个时间点距离初始时间过去了 多久
            augment dataset with age covariate
        """

        # age (timesteps from 0 for each unique time series)
        self._data['_age'] = self._data.groupby(self._grouping_name).cumcount()
        self._train_set_ages = self._data.groupby(self._grouping_name)['_age'].agg('max')
        self._train_set_ages['dummy_test_category'] = 0

    def _datetime_augmentation(
            self,
            df: pd.DataFrame
    ) -> pd.DataFrame:

        """
        util function
        增加日期相关的covariates
            augment dataset with datetime covariates
        """

        orig_cols = df.shape[1]

        # datetime features
        df['_hour_of_day'] = df[self._time_name].dt.hour
        df['_day_of_week'] = df[self._time_name].dt.dayofweek
        df['_day_of_month'] = df[self._time_name].dt.day
        df['_day_of_year'] = df[self._time_name].dt.dayofyear
        df['_week_of_year'] = df[self._time_name].dt.weekofyear
        df['_month_of_year'] = df[self._time_name].dt.month
        df['_year'] = df[self._time_name].dt.year

        # record datetime feature count
        self._datetime_feat_ct = df.shape[1] - orig_cols

        return df

    def _train_val_split(
            self,
            val_split: float
    ):
        """
        util function
            split dataset object into training and validation data frames
        """

        # split data into training and validation sets
        assert val_split >= 0 and val_split < 1, \
            'Validation split must be between 0 (inclusive) and 1 (exclusive)'

        if val_split == 0:
            self._train_data = self._data.copy()
        else:
            nrow = int(self._data.shape[0] * val_split)
            self._train_data = self._data.head(self._data.shape[0] - nrow)
            self._val_data = self._data.tail(nrow)

    def _create_sampling_dist(self):
        """
        util function
            create scaled sampling distribution over time series 
        """

        scale_factors = self._target_means

        # softmax the distribution for sampling
        e_x = np.exp(scale_factors - np.max(scale_factors))
        self._scale_factors_softmax = e_x / e_x.sum(axis=0)

    def _store_target_means(
            self,
            val_split: float
    ):
        """
        util function
            stores target means and create scaled sampling distribution according to means
        """

        # store target means over training set
        # self._target_means代表每个group下的目标值的平均值+1
        self._target_means = 1 + self._train_data.groupby(self._grouping_name)['target'].agg('mean')
        target_mean = self._train_data['target'].dropna().mean()

        # create scale factor sampling dist. before adding dummy keys
        self._create_sampling_dist()

        # add 'dummy_test_category' as key to target means
        self._target_means[self._label_encoder.transform(['dummy_test_category'])[0]] = target_mean

        # 如果在Validation中的groupby value 在训练集中不存在，使用训练集drop na后的mean填充
        if val_split != 0:
            # if group in val doesn't exist in train, standardize by overall mean
            for group in self._val_data[self._grouping_name].unique():
                if group not in self._train_data[self._grouping_name].unique():
                    self._target_means[group] = target_mean

    def _mask_missing_targets(
            self,
            df: pd.DataFrame
    ):
        """
        util function
            mask missing target values in training and validation frames
        """

        # mask missing target values
        for idx in pd.isnull(df)['target'].to_numpy().nonzero()[0]:
            key = df[self._grouping_name].iloc[idx]
            if key in self._missing_tgt_vals.keys():
                self._missing_tgt_vals[key].append(df['_age'].iloc[idx])
            else:
                self._missing_tgt_vals[key] = [df['_age'].iloc[idx]]

    def _standardize(
            self,
            val_split: float
    ):
        """ 
        util function
            standardize covariates and record locations of missing tgt values (for standardization later)
        """
        # standardize covariates N(0,1) and 'target' col by mean
        covariate_names = ['target', self._grouping_name, self._time_name]
        covariate_mask = [False if col_name in covariate_names else True for col_name in self._data.columns]
        self._scaler = StandardScaler()
        self._train_data.loc[:, covariate_mask] = self._scaler.fit_transform(
            self._train_data.loc[:, covariate_mask].astype('float'))

        # record locations of missing target values
        self._missing_tgt_vals = {}
        self._mask_missing_targets(self._train_data)

        if val_split != 0:
            self._val_data = self._val_data.reset_index(drop=True)
            self._val_data.loc[:, covariate_mask] = self._scaler.transform(
                self._val_data.loc[:, covariate_mask].astype('float'))

            # record locations of missing target values
            self._mask_missing_targets(self._val_data)

        # keep full dataset up to date
        self._data.loc[:, covariate_mask] = self._scaler.transform(self._data.loc[:, covariate_mask].astype('float'))

    def _pad_ts(
            self,
            pandas_df: pd.DataFrame,
            desired_len: int
    ) -> pd.DataFrame:

        """
        Pad time series to desired length
            :param pandas_df:
            :param desired_len: (int)
            :param min_date: pd timedelta (default None)
            :return: padded df
        """

        # TODO CHECK THAT THIS INTERPOLATES ONE HOTS CORRECTLY
        # padding_df = self._datetime_interpolation(pandas_df,
        #                                           pandas_df[self._time_name].iloc[-1],
        #                                           negative_offset=desired_len - pandas_df.shape[0]
        #                                           )
        padding_df = self._datetime_augmentation(pandas_df)

        # standardize ONLY datetime covariates N(0,1) others interpolated and thus already standardized
        covariate_names = ['target', 'prev_target', self._grouping_name, self._time_name]
        covariate_mask = [False if col_name in covariate_names else True for col_name in padding_df.columns]

        padding_df.iloc[:, -(self._datetime_feat_ct + 1):-1] = self._scaler.transform(
            padding_df.iloc[:, covariate_mask].astype('float')
        )[:, -(self._datetime_feat_ct + 1):-1]

        return padding_df

    def _sample_ts(
            self,
            pandas_df: pd.DataFrame,
            desired_len: int
    ) -> pd.DataFrame:

        """ sample window from time series from https://github.com/arrigonialberto86/deepar
        
        Arguments:
            pandas_df {pd.DataFrame} -- input pandas df with 'target' columns 
            desired_len {int} -- desired sample length (number of rows)
        
        
        Returns:
            pd.DataFrame -- df of sampled window from time series
        """

        if pandas_df.shape[0] < desired_len:
            raise ValueError('Desired sample length is greater than df row len')
        if pandas_df.shape[0] == desired_len:
            return pandas_df

        start_index = np.random.choice([i for i in range(0, pandas_df.shape[0] - desired_len)])

        return pandas_df.iloc[start_index: start_index + desired_len, ]

    def _scale_prev_target_col(
            self,
            df: pd.DataFrame,
            means: pd.Series
    ) -> pd.DataFrame:

        """
        util function
            scale previous target column by target means
        """
        if (means == 0).all():
            df.loc[:, 'prev_target'] = means
        else:
            df.loc[:, 'prev_target'] = df['prev_target'] / means

        return df

    def _add_prev_target_col(
            self,
            df: pd.DataFrame,
            target_means: pd.Series,
            train_df: pd.DataFrame = None
    ) -> pd.DataFrame:

        """
        util function
            add column with previous target value for autoregressive modeling
        """

        df = df.reset_index(drop=True)
        means = target_means[df[self._grouping_name]].reset_index(drop=True)
        if train_df is None:

            # add feature column for previous output value (z_{t-1})
            df.loc[:, 'prev_target'] = pd.Series([0]).append(df['target'].iloc[:-1], ignore_index=True)

            # scale
            df = self._scale_prev_target_col(df, means)

        else:
            # validation set
            if 'target' in df.columns:
                if train_df.shape[0] > 0:
                    df.loc[:, 'prev_target'] = train_df['target'].dropna().tail(1).append(df['target'].iloc[:-1],
                                                                                          ignore_index=True)
                else:
                    df.loc[:, 'prev_target'] = pd.Series([0]).append(df['target'].iloc[:-1], ignore_index=True)

            # test set
            else:
                df.loc[:, 'prev_target'] = \
                    train_df['target'].dropna().tail(1).repeat(repeats=df.shape[0]).reset_index(drop=True)

            # scale
            df = self._scale_prev_target_col(df, means)

        # interpolate
        df.loc[:, 'prev_target'] = df['prev_target'].interpolate(limit_direction='both')

        return df

    def _sample_missing_prev_tgts(
            self,
            df: pd.DataFrame,
            full_df_ages: pd.Series,
            model: Model,
            window_size: int,
            batch_size: int,
            training: bool = True
    ) -> pd.DataFrame:

        """ sample missing previous target values from current model parameters
        
        Arguments:
            df {pd.DataFrame} -- df for which we would like to fill missing target values
            full_df_ages {pd.Series} -- series of ages from full, unsampled df - necessary because 
                missing target could exist in previous row that wasn't sampled (needed for prev_target input)
            model {Model} -- Keras model object containing current model parameters
            window_size {int} -- length of sampled series
            batch_size {int} -- number of series in current batch
        
        Keyword Arguments:
            training {bool} -- whether we are sampling missing values during training (lstm dropout on) 
                or testing (lstm dropout off) (default: {True})
        
        Returns:
            pd.DataFrame -- df with missing previous target values sampled from current model parameters
        """

        # sample missing 'targets' from current model parameters (for 'prev_targets')
        category = df[self._grouping_name].iloc[0]
        if category in self._missing_tgt_vals.keys():

            # get indices from full df age column to check for missing values (because missing target
            # could exist in previous row that wasn't sampled)
            age_list = full_df_ages.reindex([i - 1 for i in df.index.values.tolist()])

            if not set(self._missing_tgt_vals[category]).isdisjoint(age_list) and df.shape[0] == window_size:
                drop_list = [col for col in df.columns if
                             col == self._grouping_name or col == self._time_name or col == 'target']
                cont = tf.constant(
                    np.repeat(df.drop(drop_list, 1).values.reshape(1, window_size, -1), batch_size, axis=0),
                    dtype=tf.float32)
                cat = tf.constant(np.repeat(df[self._grouping_name].values.reshape(1, window_size), batch_size, axis=0),
                                  dtype=tf.float32)
                preds = model([cont, cat], training=training)[0][0]

                # refill indices (add 1 for each negative observation, i.e. before start of series)
                refill_indices = df.index[age_list.isin(self._missing_tgt_vals[category])]
                if df.index[0] > 0:
                    refill_values = [preds[i] for i in [r - df.index[0] for r in refill_indices]]
                else:
                    refill_values = [preds[i] for i in refill_indices]
                for idx, val in zip(refill_indices, refill_values):
                    df['prev_target'][idx] = val

        return df

    def next_batch(
            self,
            model: Model,
            batch_size: int,
            window_size: int,
            verbose: bool = False,
            val_set: bool = False
    ) -> typing.Tuple[typing.List[tf.Tensor], tf.Tensor, tf.Tensor]:

        """ gets next batch of training data
        
        Arguments:
            model {Model} -- model object, allows sampling for missing target obs. in training set
            batch_size {int} -- how many time series to be sampled in this batch (int)
            window_size {int} -- window of each sampled time series
        
        Keyword Arguments:
            verbose {bool} -- (default: {False})
            val_set {bool} -- if True, will sample from validation set, if False, will sample
                from trianing set (default: {False})
        
        Returns:
            typing.Tuple[typing.List[tf.Tensor], tf.Tensor, tf.Tensor] -- 
                [X_continouous, X_categorical], 
                C (categorical grouping variable), 
                y
        """

        # Generate sampling of time series according to prob dist. defined by scale factors
        if val_set:
            assert self._val_data is not None, "Asking for validation batch, but validation split was 0 in object construction"
            # 一个batch里 随机取groupby列（categorical）的unique值，
            # 所以其实并不是每个category 变量都会被选到的，其实是随机出来的
            # 然后对于每个category的子集，某个值下的子数据集，再构建batch
            cat_samples = np.random.choice(self._val_data[self._grouping_name].unique(), batch_size)
            data = self._val_data
        else:
            cat_samples = np.random.choice(self._train_data[self._grouping_name].unique(), batch_size,
                                           p=self._scale_factors_softmax)
            data = self._train_data

        sampled = []
        for cat in cat_samples:
            # 对每个随机取到的unique值，找到对应的子数据集，然后增加prev_target_col这个选项
            cat_data = data[data[self._grouping_name] == cat]

            # add 'prev_target' column for this category
            if val_set:
                cat_data = self._add_prev_target_col(cat_data, self._target_means,
                                                     self._train_data[self._train_data[self._grouping_name] == cat])
            else:
                import pdb
                cat_data = self._add_prev_target_col(cat_data, self._target_means)
            cat_data.loc[:, 'target'] = cat_data['target'].fillna(self._mask_value)

            # sample window from time series
            if cat_data.shape[0] > window_size:
                sampled_cat_data = self._sample_ts(pandas_df=cat_data,
                                                   desired_len=window_size)
            else:
                sampled_cat_data = data

            # sample missing 'targets' from current model parameters (for 'prev_targets')
            sampled_cat_data = self._sample_missing_prev_tgts(
                sampled_cat_data,
                cat_data['_age'],
                model,
                window_size,
                batch_size
            )

            sampled.append(sampled_cat_data)
        data = pd.concat(sampled)
        # window_size默认是20  batch_size是16
        # cont_inputs 就是会加上时间相关的，比如在案例里是10个 就是 [16， 20， 10]的大小
        cont_inputs = tf.constant(
            data.drop(
                ['target', self._grouping_name, self._time_name],
                1
            ).values.reshape(batch_size, window_size, -1),
            dtype=tf.float32
        )
        cat_inputs = tf.constant(
            data[self._grouping_name].values.reshape(batch_size, window_size),
            dtype=tf.float32
        )
        cat_labels = tf.constant(
            cat_samples.reshape(batch_size, 1),
            dtype=tf.int32
        )
        targets = tf.constant(
            data['target'].values.reshape(batch_size, window_size, 1),
            dtype=tf.float32
        )
        import pdb
        # pdb.set_trace()
        return [cont_inputs, cat_inputs], cat_labels, targets

    ## names and indices
    @property
    def grouping_name(self):
        return self._grouping_name

    @property
    def time_name(self):
        return self._time_name

    @property
    def grouping_idx(self):
        return self._grouping_idx

    @property
    def timestamp_idx(self):
        return self._timestamp_idx

    @property
    def index_col(self):
        return self._index_col

    @property
    def one_hot_indices(self):
        return self._one_hot_indices

    ## constructor properties
    @property
    def negative_obs(self):
        return self._negative_obs

    @property
    def freq(self):
        return self._freq

    @property
    def mask_value(self):
        return self._mask_value

    @property
    def count_data(self):
        return self._count_data

    @property
    def normalize_dates(self):
        return self._normalize_dates

    @property
    def unique_cats(self):
        return self._unique_cats

    @property
    def num_cats(self):
        return self._num_cats

    @property
    def features(self):
        return self._features

    ## object learned features
    @property
    def train_set_ages(self):
        return self._train_set_ages

    @property
    def max_age(self):
        return self._train_set_ages.max()

    @property
    def target_means(self):
        return self._target_means

    @property
    def missing_tgt_vals(self):
        return self._missing_tgt_vals

    @property
    def scaler(self):
        return self._scaler

    @property
    def label_encoder(self):
        return self._label_encoder

    @property
    def features(self):
        return self._features


class TimeSeriesTest(TimeSeriesTrain):

    def __init__(
            self,
            train_ts_obj: TimeSeriesTrain,
            pandas_df: pd.DataFrame = None,
            target_idx: int = None
    ):
        """ prepares TimeSeriesTest data object - augments test data with covariates, standardizes test data, encodes 
                test categorical variables (for embeddings in model), asserts compatibility with TimeSeriesTrain object, 
                calculates max prediction horizion that object supports
        
        Arguments:
            train_ts_obj {TimeSeriesTrain} -- TimeSeriesTrain object defined on training / validation set
        
        Keyword Arguments:
            pandas_df {pd.DataFrame} -- df of test time series and covariates (default: {None})
            target_idx {int} -- index of target column if one exists (default: {None})
        """

        self._data = pandas_df

        # indices of special columns must be same as train object bc of scaling transformation
        self._train_ts_obj = train_ts_obj
        self._one_hot_indices = train_ts_obj.one_hot_indices
        self._timestamp_idx = train_ts_obj.timestamp_idx
        self._grouping_idx = train_ts_obj.grouping_idx
        self._grouping_name = train_ts_obj.grouping_name
        self._count_data = train_ts_obj.count_data
        self._freq = train_ts_obj.freq
        self._time_name = train_ts_obj.time_name
        self._scaler = train_ts_obj.scaler
        self._normalize_dates = train_ts_obj.normalize_dates
        self._missing_tgt_vals = train_ts_obj.missing_tgt_vals
        self._data[self.grouping_name] = self._data[self._grouping_name].astype(str)
        import pdb
        # preprocess new test data if it exists
        if self._data is not None:
            self._preprocess_new_test_data(target_idx, train_ts_obj)
        else:
            # unique_cats是训练集的category列里unique value, 现在是test_groups
            self._test_groups = train_ts_obj.unique_cats
            self._new_test_groups = []
            self._horizon = 0

        self._batch_test_data_prepared = False

    def _preprocess_new_test_data(
            self,
            target_idx: int,
            train_ts_obj: TimeSeriesTrain
    ):
        """ util function to preprocess new test data if it exists
        """

        col_names = list(self._data)
        if self._grouping_name == 'category':
            self._data['category'] = 'dummy_cat'

        # delete target column if one exists (not needed for test)
        if target_idx is not None:
            self._data = self._data.drop(col_names[target_idx], axis=1)

            # update other indices    
            if self._timestamp_idx > target_idx:
                self._timestamp_idx -= 1
            if self._grouping_idx > target_idx:
                self._grouping_idx -= 1

                # delete index column if one exists (absolute position information only available through covariates)
        if train_ts_obj.index_col is not None:
            self._data = self._data.drop(col_names[train_ts_obj.index_col], axis=1)

        # sort df by timestamp
        self._time_name = self._sort_by_timestamp(col_names)

        # age (timesteps from beginning of train set for each unique time series)
        self._test_groups = self._data[self._grouping_name].unique()
        self._new_test_groups = []
        for group in self._test_groups:
            if group not in train_ts_obj.unique_cats:
                train_ts_obj.train_set_ages[group] = 0
                self._new_test_groups.append(group)
        import pdb
        # datetime features
        max_date = self._data.groupby(self._grouping_name)[self._time_name].agg('max').max()
        min_date_test = self._train_ts_obj._data[self._time_name].max() + robust_timedelta(1, self._freq)

        # self._data = self._datetime_interpolation(self._data, max_date, min_date=min_date_test)
        self._data = self._datetime_augmentation(self._data)
        self._age_augmentation(train_ts_obj.train_set_ages)

        # compute max prediction horizon
        if self._data.shape[0] > 0:
            self._horizon = self._data.groupby(self._grouping_name)['_age'].count().max()
        else:
            self._horizon = 0

        # standardize covariates N(0,1)
        covariate_names = [self._grouping_name, self._time_name]
        covariate_mask = [False if col_name in covariate_names else True for col_name in self._data.columns]
        if self._data.shape[0] > 0:
            self._data.loc[:, covariate_mask] = self._scaler.transform(
                self._data.loc[:, covariate_mask].astype('float'))

        # assert compatibility with training TimeSeriesTrain object after processing
        assert self._data.shape[1] - 1 == train_ts_obj.features, \
            "Number of feature columns in test object must be equal to the number in train object"
        assert self._count_data == train_ts_obj.count_data, \
            "Count data boolean in test object must be equivalent to train object"

    def _sort_by_timestamp(self, *args, **kwargs):
        return super(TimeSeriesTest, self)._sort_by_timestamp(*args, **kwargs)

    def _datetime_interpolation(self, *args, **kwargs):
        return super(TimeSeriesTest, self)._datetime_interpolation(*args, **kwargs)

    def _age_augmentation(self, train_ages):
        self._data['_age'] = self._data.groupby(self._grouping_name).cumcount() + 1
        self._data['_age'] += train_ages[self._data[self._grouping_name].values].values

    def _datetime_augmentation(self, *args, **kwargs):
        return super(TimeSeriesTest, self)._datetime_augmentation(*args, **kwargs)

    def _pad_ts(self, *args, **kwargs):
        return super(TimeSeriesTest, self)._pad_ts(*args, **kwargs)

    def _scale_prev_target_col(self, *args, **kwargs):
        return super(TimeSeriesTest, self)._scale_prev_target_col(*args, **kwargs)

    def _add_prev_target_col(self, *args, **kwargs):
        return super(TimeSeriesTest, self)._add_prev_target_col(*args, **kwargs)

    def _sample_missing_prev_tgts(self, *args, **kwargs):
        return super(TimeSeriesTest, self)._sample_missing_prev_tgts(*args, **kwargs)

    def _one_hot_padding(self, *args, **kwargs):
        return super(TimeSeriesTest, self)._one_hot_padding(*args, **kwargs)

    def _prepare_batched_test_data(
            self,
            batch_size: int,
            window_size: int,
            include_all_training: bool = False,
            verbose: bool = False
    ):
        """ Split data into batches of window_size for stateful inference
        
        Arguments:
            batch_size {int} -- batch size
            window_size {int} -- window of each sampled time series
        
        Keyword Arguments:
            include_all_training {bool} -- whether to include all training data in prep of batches (default: {False})
            verbose {bool} -- (default: {False})
        """

        if include_all_training:
            max_train_age = self._train_ts_obj._data.groupby(self._train_ts_obj.grouping_name).size().max()
            if (max_train_age - self._train_ts_obj.negative_obs) % window_size != 0:
                max_train_age = ((
                                         max_train_age - self._train_ts_obj.negative_obs) // window_size) * window_size + window_size
        else:
            max_train_age = window_size

        # interpolate all training set series from same min date (that supports window size)
        self._train_ts_obj._data = self._train_ts_obj._data.groupby(self._grouping_name).apply(
            lambda df: self._pad_ts(
                df,
                max_train_age
            )
        ).reset_index(drop=True)

        # calculate # train batches
        self._train_batch_ct = max_train_age // window_size

        # calculate number of iterations -- if batch_size < number of series in test df, 
        # we need to run multiple sequential iterations through time (one for each batch)
        self._total_iterations = len(self._test_groups) // batch_size + 1
        self._iterations = 0

        data = []
        padding_value = 0

        for cat in self._test_groups:
            import pdb
            # series category doesn't exist in training set
            enc_cat = self._train_ts_obj.label_encoder.transform([cat])[0]
            if enc_cat in self._new_test_groups:
                print('this way')
                print(self._test_groups)
                print(self._new_test_groups)
                train_data = pd.DataFrame(
                    {col: padding_value for col in self._data.columns},
                    index=[i for i in range(max_train_age)]
                )
                train_data = self._add_prev_target_col(train_data, self._train_ts_obj.target_means)

            else:
                st = time.time()
                # enc_cat = self._train_ts_obj.label_encoder.transform([cat])[0]
                train_data = self._train_ts_obj._data[self._train_ts_obj._data[self._grouping_name] == enc_cat]

                # add 'prev_target' column for this series
                train_data = self._add_prev_target_col(train_data, self._train_ts_obj.target_means)

                if train_data.shape[0] < max_train_age:
                    train_data = self._pad_ts(pandas_df=train_data, desired_len=max_train_age)

            # append test data if it exists
            if self._data is not None and self._data.shape[0] > 0:

                # convert groups to ints in test data
                test_data = self._data[self._data[self._grouping_name] == cat]
                if cat in self._new_test_groups:
                    test_data[self._grouping_name] = 'dummy_test_category'
                test_data[self._grouping_name] = self._train_ts_obj.label_encoder.transform(
                    test_data[self._grouping_name])

                # add prev target column
                test_data = self._add_prev_target_col(
                    test_data,
                    self._train_ts_obj.target_means,
                    train_df=train_data
                )

                # append test data, drop 'target' col from training data
                # import pdb
                # pdb.set_trace()
                prepped_data = pd.concat(
                    [train_data.drop('target', axis=1), test_data]
                ).reset_index(drop=True)
                # prepped_data = test_data
            else:
                prepped_data = train_data
            data.append(prepped_data)

        self._prepped_data = data
        self._batch_test_data_prepared = True
        self._batch_idx = 0

    def next_batch(
            self,
            model: Model,
            batch_size: int,
            window_size: int,
            include_all_training: bool = False,
            verbose: bool = False

    ) -> typing.Tuple[typing.List[tf.Tensor], tf.Tensor, int, int]:

        """ Split data into batches of window_size
        
        Arguments:
            model {Model} -- trained Keras Model (for sampling missing prev tgt values)
            batch_size {int} -- batch_size
            window_size {int} -- window of each sampled time series
        
        Keyword Arguments:
            include_all_training {bool} -- whether to include all training data in prep of batches (default: {False})
            verbose {bool} -- (default: {False})
        
        Returns:
            typing.Tuple[typing.List[tf.Tensor], tf.Tensor, int, int] --
                [X_continouous, X_categorical], 
                C (categorical grouping variable), 
                prediction_horizon_index, 
                iteration_index
        """

        if not self._batch_test_data_prepared:
            self._prepare_batched_test_data(batch_size,
                                            window_size,
                                            include_all_training=include_all_training,
                                            verbose=verbose)

        # if done with full sequence through time for this batch
        if self._batch_idx == self._horizon + self._train_batch_ct:
            self._iterations += 1
            self._batch_idx = 0

        # return None if no more iterations
        if self._iterations >= self._total_iterations and self._iterations > 0:
            return (None, None, None, None)

        # grab current batch
        df_start_idx = self._iterations * batch_size
        df_stop_idx = (self._iterations + 1) * batch_size
        if self._batch_idx >= self._train_batch_ct:
            batch_start_idx = self._train_batch_ct * window_size + self._batch_idx - self._train_batch_ct
        else:
            batch_start_idx = self._batch_idx * window_size
        batch_data = [
            df.iloc[batch_start_idx:batch_start_idx + window_size, :]
            for df in self._prepped_data[df_start_idx:df_stop_idx]
        ]

        # sample missing 'targets' from current model parameters (for 'prev_targets')
        batch_data = [
            self._sample_missing_prev_tgts(
                b_data,
                full_df['_age'],
                model,
                window_size,
                batch_size,
                training=False
            )
            for b_data, full_df in zip(batch_data, self._prepped_data)
        ]

        # in case not enough rows in last batch (batches in test case) to support all indices
        batch_data = [
            b_data.append(
                pd.concat(
                    [b_data.iloc[-1:, :]] * (window_size - b_data.shape[0]),
                    ignore_index=True)
            )
            if window_size - b_data.shape[0] > 0
            else b_data
            for b_data in batch_data
        ]

        batch_df = pd.concat(batch_data)
        self._batch_idx += 1

        # prep continuous and categorical values
        drop_list = [i for i in batch_df.columns if i in [self._grouping_name, self._time_name, 'target']]
        x_cont = batch_df.drop(drop_list, 1).values.reshape(len(batch_data), window_size, -1)
        x_cat = batch_df[self._grouping_name].values.reshape(len(batch_data), window_size)
        x_cat_key = tf.constant(x_cat[:, :1], dtype=tf.int32)

        # pad data to batch size if necessary 
        if len(batch_data) < batch_size:
            x_cont = np.append(x_cont, [x_cont[0]] * (batch_size - len(batch_data)), axis=0)
            x_cat = np.append(x_cat, [x_cat[0]] * (batch_size - len(batch_data)), axis=0)
        x_cont = tf.Variable(x_cont, dtype=tf.float32)
        x_cat = tf.constant(x_cat, dtype=tf.float32)

        return ([x_cont, x_cat],
                x_cat_key,
                self._batch_idx - self._train_batch_ct,
                self._iterations)

        ## inference objects

    @property
    def horizon(self):
        return self._horizon

    @property
    def test_groups(self):
        return self._test_groups

    ## objects that we sometimes need to set
    @property
    def batch_idx(self):
        return self._batch_idx

    @batch_idx.setter
    def batch_idx(self, value):
        self._batch_idx = value

    @property
    def iterations(self):
        return self._iterations

    @iterations.setter
    def iterations(self, value):
        self._iterations = value

    @property
    def batch_test_data_prepared(self):
        return self._batch_test_data_prepared

    @batch_test_data_prepared.setter
    def batch_test_data_prepared(self, value):
        self._batch_test_data_prepared = value


def train_ts_generator(
        model: Model,
        ts_obj: TimeSeriesTrain,
        batch_size: int,
        window_size: int,
        verbose: bool = False,
        val_set: bool = False
) -> typing.Tuple[typing.List[tf.Tensor], tf.Tensor, tf.Tensor]:
    """ This is a util generator function for a TimeSeriesTrain object

        Arguments:
            model {Model} -- trained Keras Model (for sampling missing prev tgt values)
            ts_obj {TimeSeriesTrain} -- TimeSeriesTrain ts object 
            batch_size {int} -- batch_size
            window_size {int} -- window of each sampled time series
        
        Keyword Arguments:
            verbose {bool} -- (default: {False})
            val_set {bool} -- if True, will sample from validation set, if False, will sample
                from trianing set (default: {False})

        bootstrapped from https://github.com/arrigonialberto86/deepar

    Yields:
        typing.Tuple[typing.List[tf.Tensor], tf.Tensor, tf.Tensor] -- 
            [X_continouous, X_categorical], 
            C (categorical grouping variable), 
            y
    """

    while 1:
        yield ts_obj.next_batch(model,
                                batch_size,
                                window_size,
                                verbose=verbose,
                                val_set=val_set)


def test_ts_generator(
        model: Model,
        ts_obj: TimeSeriesTest,
        batch_size: int,
        window_size: int,
        include_all_training: bool = False,
        verbose: bool = False
) -> typing.Tuple[typing.List[tf.Tensor], tf.Tensor, int, int]:
    """ This is a util generator function for a TimeSeriesTest object

        Arguments:
            model {Model} -- trained Keras Model (for sampling missing prev tgt values)
            ts_obj {TimeSeriesTrain} -- TimeSeriesTest ts object 
            batch_size {int} -- batch_size
            window_size {int} -- window of each sampled time series
        
        Keyword Arguments:
            include_all_training {bool} -- whether to include all training data in prep of batches (default: {False})
            verbose {bool} -- (default: {False})

        bootstrapped from https://github.com/arrigonialberto86/deepar

    Yields:
        typing.Tuple[typing.List[tf.Tensor], tf.Tensor, int, int] --
            [X_continouous, X_categorical], 
            C (categorical grouping variable), 
            prediction_horizon_index, 
            iteration_index
    """

    while 1:
        yield ts_obj.next_batch(model,
                                batch_size,
                                window_size,
                                include_all_training=include_all_training,
                                verbose=verbose)
