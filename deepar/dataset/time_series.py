from deepar.dataset import Dataset
from deepar.dataset.utils import robust_timedelta, robust_reindex
import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
import time
import logging

logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)

pd.options.mode.chained_assignment = None  # default='warn'

class TimeSeriesTrain(Dataset):

    def __init__(self, pandas_df, target_idx = None, timestamp_idx = None, grouping_idx=None, one_hot_indices=None, 
                 index_col = None, count_data=False, negative_obs = 1, val_split = 0.2, integer_timestamps = False, 
                 freq = 'S'):
        """
            prepares TimeSeriesTrain data object - augments data with covariates, standardizes data, encodes 
                categorical variables (for embeddings in model), optionally splits into training and validation sets

            :param pandas_df: df to sample time series from
            :param target_idx: index of target column, if None Error will be raised
            :param timestamp_idx: index of column containing timestamps, timestamps are parsed as # of seconds
            :param grouping_idx: index of grouping column
            :param one_hot_indices: list of indices of columns that are one hot encoded
            :param index_col: index column, if not None will be dropped
            :param count_data: boolean indicating whether data is count data (determines loss function)
            :param negative_obs: how far before beginning of time series is it possible to set t
            :param val_split: proportion of data to withhold for validation
            :param integer_timestamps: whether timestamp column is expressed in ints (instead of float seconds)
            :param freq: frequency of the time series, default is seconds
        """
        super().__init__()

        if pandas_df is None:
            raise ValueError('Must provide a Pandas df to instantiate this class')

        # store constructor arguments as instance variables
        self.data = pandas_df
        self.integer_timestamps = integer_timestamps
        self.one_hot_indices = one_hot_indices
        self.negative_obs = negative_obs
        col_names = list(self.data)
        if grouping_idx is None:
            self.data['category'] = 'dummy_cat'
            self.grouping_name = 'category'
            self.grouping_idx = len(col_names)
        else:
            self.grouping_name = col_names[grouping_idx]
            self.grouping_idx = grouping_idx
        self.timestamp_idx = timestamp_idx
        self.count_data = count_data
        self.freq = freq

        # set name of target variable
        self.target_idx = target_idx
        if target_idx is None:
            raise ValueError('Must provide an index for a target column to instantiate this class')
        self.data = self.data.rename(columns={col_names[target_idx]:'target'})

        # set mask value
        if count_data:
            self.mask_value = 0
        else:
            self.mask_value = self.data['target'].min() - 1

        # delete index column if one exists (absolute position information only available through covariates)
        self.index_col = index_col
        if index_col is not None:
            self.data = self.data.drop(col_names[index_col], axis=1)

        # augment dataset with covariates
        self.time_name = self._sort_by_timestamp(col_names)
        self.data = self._datetime_interpolation(self.data, self.data[self.time_name].iloc[-1], negative_offset = self.negative_obs)
        self.data = self._datetime_augmentation(self.data)
        self._age_augmentation()

        # need embeddings for cats in val, even if not in train
        # 1 extra for test cats not included in train or val
        self.unique_cats = self.data[self.grouping_name].unique()
        self.num_cats = len(self.unique_cats) + 1

        # convert groups to ints
        self.label_encoder = LabelEncoder()
        cat_names = self.data[self.grouping_name].append(pd.Series(['dummy_test_category']))
        self.label_encoder.fit(cat_names)
        self.data[self.grouping_name] = self.label_encoder.transform(self.data[self.grouping_name])

        # split into train + validation sets, create sampling dist.
        self._train_val_split(val_split)
        self._store_target_means(val_split)

        # standardize
        self._standardize(val_split)

        # store number of features and categorical count and target means
        self.features = self.data.shape[1] - 2 # -3 :target, grouping name, datetime, +1 :prev target
        self.count_data = count_data

    def _sort_by_timestamp(self, col_names):
        """
        util function
            sort df by timestamp
        """
        if self.timestamp_idx is None:
            raise ValueError('Must provide the index of the timestamp column to instantiate this class')
        time_name = col_names[self.timestamp_idx]
        self.data = self.data.sort_values(by = time_name)
        if self.integer_timestamps:
            self.data[time_name] = pd.to_datetime(self.data[time_name] - 1, unit = 'D')
            self.freq = 'D'
        else:
            self.data[time_name] = pd.to_datetime(self.data[time_name], unit = 's')
        return time_name

    def _datetime_interpolation(self, df, max_date, min_date = None, negative_offset = 0):
        """
        util function
            interpolate along time dimension (to max_date) to create evenly spaced observations.
            this will interpolate covariates
        """

        # interpolate all series to max_date 
        new_dfs = []
        for group, df in df.groupby(self.grouping_name):
            if min_date is None:
                min_date_group = df[self.time_name].iloc[0] - robust_timedelta(negative_offset, self.freq)
            else:
                min_date_group = min_date

            # average duplicate timestamps before reindexing
            if sum(df[self.time_name].duplicated()) > 0:
                df = df.groupby(self.time_name).mean()
                df.insert(self.grouping_idx - 1, self.grouping_name, group)
                
            else:
                df.index = df[self.time_name]
                df = df.drop(self.time_name, axis=1)

            # reindex
            df_new = robust_reindex(df, min_date_group, max_date, freq = self.freq)
            df_new.insert(self.timestamp_idx, self.time_name, df_new.index.get_level_values(-1))
            df_new = df_new.reset_index(drop = True)

            # interpolate non-target columns
            if df.shape[0] > 1:
                df_new.loc[:, df_new.columns != 'target'] = df_new.loc[:, df_new.columns != 'target'].interpolate().ffill().bfill()
            else:
                # edge case when new df only has one row
                replace = [col for col in df_new.columns if col != self.time_name]
                df_new.loc[:, replace] = df[replace].values[0]

            new_dfs.append(df_new)  
        return pd.concat(new_dfs)

    def _age_augmentation(self):
        """
        util function
            augment dataset with age covariate
        """

        # age (timesteps from 0 for each unique time series)
        self.data['age'] = self.data.groupby(self.grouping_name).cumcount()
        self.train_set_ages = self.data.groupby(self.grouping_name)['age'].agg('max')
        self.train_set_ages['dummy_test_category'] = 0

    def _datetime_augmentation(self, df):
        """
        util function
            augment dataset with datetime covariates
        """
        # datetime features
        df['_hour_of_day'] = df[self.time_name].dt.hour
        df['_day_of_week'] = df[self.time_name].dt.dayofweek
        df['_day_of_month'] = df[self.time_name].dt.day
        df['_day_of_year'] = df[self.time_name].dt.dayofyear
        df['_week_of_year'] = df[self.time_name].dt.weekofyear
        df['_month_of_year'] = df[self.time_name].dt.month
        df['_year'] = df[self.time_name].dt.year
        return df
        
    def _train_val_split(self, val_split):
        """
        util function
            split dataset object into training and validation data frames
        """
        # split data into training and validation sets
        assert val_split >= 0 and val_split < 1, \
            'Validation split must be between 0 (inclusive) and 1 (exclusive)'

        if val_split == 0:
            self.train_data = self.data.copy()
        else:
            nrow = int(self.data.shape[0] * val_split)
            self.train_data = self.data.head(self.data.shape[0] - nrow)
            self.val_data = self.data.tail(nrow)

    def _create_sampling_dist(self):
        """
        util function
            create scaled sampling distribution over time series 
        """

        self.scale_factors = 1 + self.target_means
        self.scale_factors = self.scale_factors.apply(lambda x: max(0, x))

        # softmax the distribution for sampling
        e_x = np.exp(self.scale_factors - np.max(self.scale_factors))
        self.scale_factors_softmax = e_x / e_x.sum(axis = 0)

    def _store_target_means(self, val_split):
        """
        util function
            stores target means and create scaled sampling distribution according to means
        """

        # store target means over training set
        self.target_means = self.train_data.groupby(self.grouping_name)['target'].agg('mean')
        self.target_mean = self.train_data['target'].dropna().mean()

        # create scale factor sampling dist. before adding dummy keys
        self._create_sampling_dist()

        # add 'dummy_test_category' as key to target means
        self.target_means[self.label_encoder.transform(['dummy_test_category'])[0]] = self.target_mean

        if val_split != 0:
            # if group in val doesn't exist in train, standardize by overall mean
            for group in self.val_data[self.grouping_name].unique():
                if group not in self.train_data[self.grouping_name].unique():
                    self.target_means[group] = self.target_mean

    def _mask_missing_targets(self, df):
        """
        util function
            mask missing target values in training and validation frames
        """
        # mask missing target values
        for idx in pd.isnull(df)['target'].to_numpy().nonzero()[0]:
            key = df[self.grouping_name].iloc[idx]
            if key in self.missing_tgt_vals.keys():
                self.missing_tgt_vals[key].append(df['age'].iloc[idx])
            else:
                self.missing_tgt_vals[key] = [df['age'].iloc[idx]]

    def _standardize(self, val_split):
        """ 
        util function
            standardize covariates and record locations of missing tgt values (for standardization later)
        """
        # standardize covariates N(0,1) and 'target' col by mean
        covariate_mask = [False if col_name == 'target' or col_name == self.grouping_name or col_name == self.time_name
             else True for col_name in self.data.columns]
        self.scaler = StandardScaler()
        self.train_data.loc[:, covariate_mask] = self.scaler.fit_transform(self.train_data.loc[:, covariate_mask].astype('float'))
        
        # record locations of missing target values
        self.missing_tgt_vals = {}
        self._mask_missing_targets(self.train_data)

        if val_split != 0:
            self.val_data = self.val_data.reset_index(drop=True)
            self.val_data.loc[:, covariate_mask] = self.scaler.transform(self.val_data.loc[:, covariate_mask].astype('float'))

            # record locations of missing target values
            self._mask_missing_targets(self.val_data)
        
        # keep full dataset up to date
        self.data.loc[:, covariate_mask] = self.scaler.transform(self.data.loc[:, covariate_mask].astype('float'))

    def _pad_ts(self, pandas_df, desired_len):
        """
        Pad time series to desired length
            :param pandas_df:
            :param desired_len: (int)
            :param min_date: pd timedelta (default None)
            :return: padded df
        """

        # TODO CHECK THAT THIS INTERPOLATES ONE HOTS CORRECTLY
        padding_df = self._datetime_interpolation(pandas_df, 
            pandas_df[self.time_name].iloc[-1], 
            negative_offset = desired_len - pandas_df.shape[0]
        )
        padding_df = self._datetime_augmentation(padding_df)
        
        # standardize ONLY datetime covariates N(0,1) others interpolated and thus already standardized
        covariate_mask = [False if col_name == 'target' or col_name == 'prev_target' or col_name == self.grouping_name 
            or col_name == self.time_name else True for col_name in padding_df.columns]
        padding_df.iloc[:, -8:-1] = self.scaler.transform(padding_df.iloc[:, covariate_mask].astype('float'))[:, -8:-1]

        return padding_df

    def _sample_ts(self, pandas_df, desired_len):
        """
            :param pandas_df: input pandas df with 'target' columns e features
            :param desired_len: desired sample length (number of rows)
            :return: a pandas df (sample)
        
        from https://github.com/arrigonialberto86/deepar
        """
        if pandas_df.shape[0] < desired_len:
            raise ValueError('Desired sample length is greater than df row len')
        if pandas_df.shape[0] == desired_len:
            return pandas_df

        start_index = np.random.choice([i for i in range(0, pandas_df.shape[0] - desired_len)])

        return pandas_df.iloc[start_index: start_index+desired_len, ]

    def _scale_prev_target_col(self, df, means):
        """
        util function
            scale previous target column by target means
        """
        if (means == 0).all():
            df.loc[:, 'prev_target'] = means
        else:
            df.loc[:, 'prev_target'] = df['prev_target'] / means
        
        return df

    def _add_prev_target_col(self, df, target_means, train_df = None):
        """
        util function
            add column with previous target value for autoregressive modeling
        """

        df = df.reset_index(drop=True)
        means = target_means[df[self.grouping_name]].reset_index(drop = True)
        if train_df is None:

            # add feature column for previous output value (z_{t-1})
            df.loc[:,'prev_target'] = pd.Series([0]).append(df['target'].iloc[:-1], ignore_index=True)

            # scale
            df = self._scale_prev_target_col(df, means)

        else:
            # validation set
            if 'target' in df.columns:
                if train_df.shape[0] > 0:
                    df.loc[:, 'prev_target'] = train_df['target'].dropna().tail(1).append(df['target'].iloc[:-1], ignore_index=True)
                else:
                    df.loc[:,'prev_target'] = pd.Series([0]).append(df['target'].iloc[:-1], ignore_index=True)

            # test set
            else:
                df.loc[:, 'prev_target'] = \
                    train_df['target'].dropna().tail(1).repeat(repeats = df.shape[0]).reset_index(drop = True)

            # scale
            df = self._scale_prev_target_col(df, means)

        # interpolate
        df.loc[:,'prev_target'] = df['prev_target'].interpolate(limit_direction = 'both')

        return df

    def _sample_missing_tgts(self, df, full_df, model, category, missing_tgt_vals, window_size, batch_size):
        """
        util function
            sample missing target values from current model parameters
        """

        # sample missing 'targets' from current model parameters (for 'prev_targets')
        if category in missing_tgt_vals.keys():
            
            # get indices from full df to check for missing values (because missing target
            # could exist in previous row that wasn't sampled)
            age_list = full_df['age'].reindex([i - 1 for i in df.index.values.tolist()])

            if not set(missing_tgt_vals[category]).isdisjoint(age_list) and df.shape[0] == window_size:
                drop_list = [col for col in df.columns if col == self.grouping_name or col == self.time_name or col == 'target']
                cont = tf.constant(np.repeat(df.drop(drop_list, 1).values.reshape(1, window_size, -1), batch_size, axis = 0), dtype = tf.float32)
                cat = tf.constant(np.repeat(df[self.grouping_name].values.reshape(1, window_size), batch_size, axis = 0), dtype = tf.float32)
                preds = model([cont, cat], training = True)[0][0]

                # refill indices (add 1 for each negative observation, i.e. before start of series)
                refill_indices = df.index[age_list.isin(missing_tgt_vals[category])]
                if df.index[0] > 0:
                    refill_values = [preds[i] for i in [r - df.index[0] for r in refill_indices]]
                else:
                    refill_values = [preds[i] for i in refill_indices]
                for idx, val in zip(refill_indices, refill_values):
                    df['prev_target'][idx] = val

        return df

    def next_batch(self, model, batch_size, window_size, verbose=False, val_set = False):
        """
            gets next batch of training data

            :param model: model object, allows sampling for missing target obs. in training set
            :param batch_size: how many time series to be sampled in this batch (int)
            :param window_size: window of each sampled time series
            :param verbose: default false
            :param val_set: boolean, whether this generator should sample for training or validation
            :return: [X_continouous, X_categorical], C (categorical grouping variable), y

            bootstrapped from https://github.com/arrigonialberto86/deepar
        """

        # Generate sampling of time series according to prob dist. defined by scale factors
        if val_set:
            assert self.val_data is not None, "Asking for validation batch, but validation split was 0 in object construction"
            cat_samples = np.random.choice(self.val_data[self.grouping_name].unique(), batch_size)
            data = self.val_data
        else:
            cat_samples = np.random.choice(self.train_data[self.grouping_name].unique(), batch_size, 
                p = self.scale_factors_softmax)
            data = self.train_data

        sampled = []
        for cat in cat_samples:
            cat_data = data[data[self.grouping_name] == cat]

            # add 'prev_target' column for this category
            if val_set:
                cat_data = self._add_prev_target_col(cat_data, self.target_means, 
                    self.train_data[self.train_data[self.grouping_name] == cat])
            else:
                cat_data = self._add_prev_target_col(cat_data, self.target_means)
            cat_data.loc[:, 'target'] = cat_data['target'].fillna(self.mask_value)

            # TODO problems w/ phem above window size 20
            # Initial padding for each selected time series to reach window_size
            if cat_data.shape[0] < window_size:
                sampled_cat_data = self._pad_ts(pandas_df=cat_data,
                                                desired_len=window_size)       
                if sampled_cat_data.shape[0] > window_size:
                    logger.debug(f'Shape of too big df: {sampled_cat_data.shape}')
                    sampled_cat_data = self._sample_ts(pandas_df=sampled_cat_data,
                                                        desired_len=window_size)  
                    logger.debug(f'Shape of fixed? df: {sampled_cat_data.shape}')   
            # sample window from time series
            if cat_data.shape[0] > window_size:
                sampled_cat_data = self._sample_ts(pandas_df=cat_data,
                                                    desired_len=window_size)

            # sample missing 'targets' from current model parameters (for 'prev_targets')
            sampled_cat_data = self._sample_missing_tgts(sampled_cat_data, cat_data, model, cat, self.missing_tgt_vals, window_size, batch_size)

            sampled.append(sampled_cat_data)
        data = pd.concat(sampled)
        
        # [cont_inputs, cat_inputs], cat_labels, targets
        return ([tf.constant(data.drop(['target', self.grouping_name, self.time_name], 1).values.reshape(batch_size, window_size, -1), dtype = tf.float32),
               tf.constant(data[self.grouping_name].values.reshape(batch_size, window_size), dtype = tf.float32)], 
               tf.constant(cat_samples.reshape(batch_size, 1), dtype = tf.int32),
               tf.constant(data['target'].values.reshape(batch_size, window_size, 1), dtype = tf.float32))

class TimeSeriesTest(TimeSeriesTrain):

    def __init__(self, train_ts_obj, pandas_df = None, mask_value = 0, target_idx = None):
        """
            prepares TimeSeriesTest data object - augments test data with covariates, standardizes test data, encodes 
                test categorical variables (for embeddings in model), asserts compatibility with TimeSeriesTrain object, 
                calculates max prediction horizion that object supports

            :param train_ts_obj: TimeSeries object defined on training / validation set
            :param pandas_df: df of test time series
            :param mask_value: mask to use on testing timestep indices > 1 (bc predictions batched)
            :param target_idx: index of target column
        """
        
        self.data = pandas_df

        # indices of special columns must be same as train object bc of scaling transformation
        self.train_ts_obj = train_ts_obj
        self.one_hot_indices = train_ts_obj.one_hot_indices
        self.timestamp_idx = train_ts_obj.timestamp_idx
        self.grouping_idx = train_ts_obj.grouping_idx
        self.grouping_name = train_ts_obj.grouping_name
        self.integer_timestamps = train_ts_obj.integer_timestamps
        self.count_data = train_ts_obj.count_data
        self.freq = train_ts_obj.freq
        self.mask_value = mask_value
        self.time_name = train_ts_obj.time_name
        self.scaler = train_ts_obj.scaler

        # preprocess new test data if it exists
        if self.data is not None:
            self._preprocess_new_test_data(target_idx, train_ts_obj)
        else:
            self.test_groups = train_ts_obj.unique_cats
            self.new_test_groups = []
            self.horizon = 0

        self.batch_test_data_prepared = False

    def _preprocess_new_test_data(self, target_idx, train_ts_obj):
        """ util function to preprocess new test data if it exists
        """

        col_names = list(self.data)
        if self.grouping_name == 'category':
            self.data['category'] = 'dummy_cat'

        # delete target column if one exists (not needed for test)
        if target_idx is not None:
            self.data = self.data.drop(col_names[target_idx], axis=1)

        # delete index column if one exists (absolute position information only available through covariates)
        if train_ts_obj.index_col is not None:
            self.data = self.data.drop(col_names[train_ts_obj.index_col], axis=1)

        # sort df by timestamp
        self.time_name = self._sort_by_timestamp(col_names)

        # age (timesteps from beginning of train set for each unique time series)
        self.test_groups = self.data[self.grouping_name].unique()
        self.new_test_groups = []
        for group in self.test_groups:
            if group not in train_ts_obj.unique_cats:
                train_ts_obj.train_set_ages[group] = 0
                self.new_test_groups.append(group)

        # datetime features   
        max_date = self.data.groupby(self.grouping_name)[self.time_name].agg('max').max()
        min_date_test = self.train_ts_obj.data[self.time_name].max() + robust_timedelta(1, self.freq)
        
        self.data = self._datetime_interpolation(self.data, max_date, min_date = min_date_test)
        self.data = self._datetime_augmentation(self.data)
        self._age_augmentation(train_ts_obj.train_set_ages)

        # compute max prediction horizon
        if self.data.shape[0] > 0:
            self.horizon = self.data.groupby(self.grouping_name)['age'].count().max()
        else:
            self.horizon = 0
        
        # standardize covariates N(0,1)
        covariate_mask = [False if col_name == self.grouping_name or col_name == self.time_name
            else True for col_name in self.data.columns]
        if self.data.shape[0] > 0:
            self.data.loc[:, covariate_mask] = self.scaler.transform(self.data.loc[:, covariate_mask].astype('float'))

        # assert compatibility with training TimeSeriesTrain object after processing
        assert self.data.shape[1] - 1 == train_ts_obj.features, \
            "Number of feature columns in test object must be equal to the number in train object"
        assert self.count_data == train_ts_obj.count_data, \
            "Count data boolean in test object must be equivalent to train object"

    def _sort_by_timestamp(self, *args, **kwargs):
        return super(TimeSeriesTest, self)._sort_by_timestamp(*args, **kwargs)

    def _datetime_interpolation(self, *args, **kwargs):
        return super(TimeSeriesTest, self)._datetime_interpolation(*args, **kwargs)

    def _age_augmentation(self, train_ages):
        self.data['age'] = self.data.groupby(self.grouping_name).cumcount() + 1
        self.data['age'] += train_ages[self.data[self.grouping_name].values].values

    def _datetime_augmentation(self, *args, **kwargs):
        return super(TimeSeriesTest, self)._datetime_augmentation(*args, **kwargs)

    def _pad_ts(self, *args, **kwargs):
        return super(TimeSeriesTest, self)._pad_ts(*args, **kwargs)

    def _scale_prev_target_col(self, *args, **kwargs):
        return super(TimeSeriesTest, self)._scale_prev_target_col(*args, **kwargs)

    def _add_prev_target_col(self, *args, **kwargs):
        return super(TimeSeriesTest, self)._add_prev_target_col(*args, **kwargs)

    def _sample_missing_tgts(self, *args, **kwargs):
        return super(TimeSeriesTest, self)._sample_missing_tgts(*args, **kwargs)

    def _prepare_batched_test_data(self, batch_size, window_size, include_all_training = False, padding_value = 0, verbose = False):
        """
        Split data into batches of window_size (all batches include all categories) for stateful inference
            :param batch_size: batch size
            :param window_size: window of each sampled time series
            :param include_all_training: whether to include all training data in prep of batches
            :param padding_value: default 0
            :param verbose: default false
        """

        if include_all_training:
            max_train_age = self.train_ts_obj.data.groupby(self.train_ts_obj.grouping_name).size().max()
            if (max_train_age - self.train_ts_obj.negative_obs) % window_size != 0:
                max_train_age = ((max_train_age - self.train_ts_obj.negative_obs) // window_size) * window_size + window_size
        else:
            max_train_age = window_size

        # interpolate all training set series from same min date (that supports window size)
        self.train_ts_obj.data = self.train_ts_obj.data.groupby(self.grouping_name).apply(lambda df: self._pad_ts(df, max_train_age)).reset_index(drop = True)

        # calculate # train batches
        self.train_batch_ct = max_train_age // window_size
        logger.debug(f'train batch ct: {self.train_batch_ct}')

        data = []
        self.scale_keys = []
        logger.debug(f'new test groups: {self.new_test_groups}')
        for cat in self.test_groups:
            if cat in self.new_test_groups:
                train_data = pd.DataFrame({col: self.train_ts_obj.padding_value for col in self.data.columns}, 
                    index=[i for i in range(max_train_age)])
                
                # add 'prev_target' column for this series
                train_data = self._add_prev_target_col(train_data, self.train_ts_obj.target_means)
            
            else:
                enc_cat = self.train_ts_obj.label_encoder.transform([cat])[0]

                train_data = self.train_ts_obj.data[self.train_ts_obj.data[self.grouping_name] == enc_cat]

                # add 'prev_target' column for this series
                train_data = self._add_prev_target_col(train_data, self.train_ts_obj.target_means)

                if train_data.shape[0] < max_train_age:
                    train_data = self._pad_ts(pandas_df=train_data, desired_len=max_train_age)

            # append test data if it exists
            if self.data is not None and self.data.shape[0] > 0:

                # convert groups to ints in test data
                test_data = self.data[self.data[self.grouping_name] == cat]
                if cat in self.new_test_groups:
                    test_data[self.grouping_name] = 'dummy_test_category'
                test_data[self.grouping_name] = self.train_ts_obj.label_encoder.transform(test_data[self.grouping_name])
                
                # add prev target column
                test_data = self._add_prev_target_col(test_data, self.train_ts_obj.target_means, train_df = train_data)

                # append test data, drop 'target' col from training data
                prepped_data = \
                    pd.concat([train_data.drop('target', axis=1), test_data]).reset_index(drop = True)
            else:
                prepped_data = train_data
            data.append(prepped_data)
            
            # track scale keys for inverse scaling at inference
            self.scale_keys.append(prepped_data[self.grouping_name][0])

        self.prepped_data = data
        self.batch_test_data_prepared = True
        self.batch_idx = 0

        # pad scale keys until batch size
        self.scale_keys.extend([self.scale_keys[0]] * (batch_size - len(self.scale_keys)))
        self.scale_keys = \
            tf.constant(np.array(self.scale_keys).reshape(batch_size, 1), dtype = tf.int32)

    def _one_hot_padding(self, *args, **kwargs):
        return super(TimeSeriesTest, self)._one_hot_padding(*args, **kwargs)

    def next_batch(self, model, batch_size, window_size, include_all_training = False, padding_value = 0, verbose=False):
        """
        Split data into batches of window_size (all batches include all categories) for stateful inference

            :param batch_size: batch size
            :param window_size: window of each sampled time series
            :param include_all_training: whether to include all training data in prep of batches
            :param padding_value: default 0
            :param verbose: default false
            :return [X_continouous, X_categorical], C (categorical grouping variable), prediction_horizon_index
        """

        if not self.batch_test_data_prepared:
            self._prepare_batched_test_data(batch_size, 
                window_size, 
                include_all_training = include_all_training, 
                verbose = verbose)

        # return None if no more batches
        if self.batch_idx == self.horizon + self.train_batch_ct:
            return (None, None)
        
        # grab current batch
        if self.batch_idx >= self.train_batch_ct:
            start_idx = self.train_batch_ct * window_size + self.batch_idx - self.train_batch_ct
        else:
            start_idx = self.batch_idx * window_size
        batch_data = [df.iloc[start_idx:start_idx + window_size, :] for df in self.prepped_data]
        logger.debug(f'batch i: {self.batch_idx}, start idx: {start_idx}')

        # sample missing 'targets' from current model parameters (for 'prev_targets')
        batch_data = [self._sample_missing_tgts(b_data, full_df, model, b_data[self.grouping_name].iloc[0], 
            self.train_ts_obj.missing_tgt_vals, window_size, batch_size) 
            for b_data, full_df in zip(batch_data, self.prepped_data)]

        # in case not enough rows in last batch (batches in test case) to support all indices
        batch_data = [b_data.append(pd.concat([b_data.iloc[-1:, :]] * (window_size - b_data.shape[0]), 
            ignore_index = True)) if window_size - b_data.shape[0] > 0 else b_data for b_data in batch_data]

        batch_data = pd.concat(batch_data)
        self.batch_idx += 1

        drop_list = [i for i in batch_data.columns if i == self.grouping_name or i == self.time_name or i == 'target']
        x_cont = batch_data.drop(drop_list, 1).values.reshape(len(self.test_groups), window_size, -1)
        x_cat = batch_data[self.grouping_name].values.reshape(len(self.test_groups), window_size)
        x_cont = tf.Variable(np.append(x_cont, [x_cont[0]] * (batch_size - len(self.test_groups)), axis = 0), 
            dtype = tf.float32)
        x_cat = tf.constant(np.append(x_cat, [x_cat[0]] * (batch_size - len(self.test_groups)), axis = 0), 
            dtype = tf.float32)
        logger.debug(f'horizon idx: {self.batch_idx - self.train_batch_ct}')
        return ([x_cont, x_cat], self.batch_idx - self.train_batch_ct) 

def train_ts_generator(model, ts_obj, batch_size, window_size, verbose = False, val_set = False):
    """
    This is a util generator function
        :param model: model (with current parameters) to sample missing tgt values
        :param ts_obj: a TimeSeries class object that implements the 'next_batch' method
        :param batch_size: batch size
        :param window_size: window of each sampled time series
        :param verbose: default false
        :param val_set: boolean for mode, True is validation 
        :yield: [X_continouous, X_categorical], C (categorical grouping variable), y

    bootstrapped from https://github.com/arrigonialberto86/deepar
    """
    while 1:
        yield ts_obj.next_batch(model, 
                batch_size, 
                window_size, 
                verbose = verbose, 
                val_set = val_set)

def test_ts_generator(model, ts_obj, batch_size, window_size, include_all_training = False, padding_value = 0, verbose = False):
    """
    This is a util generator function
        :param model: model (with current parameters) to sample missing tgt values
        :param ts_obj: a TimeSeriesTest class object that implements the 'next_batch' method
        :param batch_size: batch size
        :param window_size: window of each sampled time series
        :param include_all_training: whether to start inference at beginning of training set
        :param padding_value: default 0
        :param verbose: default false
        :yield: [X_continouous, X_categorical], C (categorical grouping variable), prediction_horizon_index
    """
    
    while 1:
        x_test, horizon_idx = ts_obj.next_batch(model, 
            batch_size, 
            window_size, 
            include_all_training = include_all_training,
            padding_value = padding_value,
            verbose = verbose)
        x_test, horizon_idx
        yield x_test, horizon_idx


