from deepar.dataset import Dataset
import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf

pd.options.mode.chained_assignment = None  # default='warn'

class TimeSeries(Dataset):

    def __init__(self, pandas_df, target_idx = None, timestamp_idx = None, grouping_idx=None, one_hot_indices=None, 
                 index_col = None, count_data=False, negative_obs = 1, val_split = 0.2, mask_value = -10000):
        """
            :param pandas_df: df to sample time series from
            :param target_idx: index of target column, if None Error will be raised
            :param timestamp_idx: index of column containing timestamps, timestamps are parsed as # of seconds
            :param grouping_idx: index of grouping column
            :param one_hot_indices: list of indices of columns that are one hot encoded
            :param index_col: index column, if not None will be dropped
            :param count_data: boolean indicating whether data is count data (determines loss function)
            :param negative_obs: how far before beginning of time series is it possible to set t
            :param val_split: proportion of data to withhold for validation
            :param mask_value: mask to use on missing target values
        """
        super().__init__()

        self.data = pandas_df
        self.one_hot_indices = one_hot_indices
        self.negative_obs = negative_obs
        col_names = list(self.data)
        if grouping_idx is None:
            self.data['category'] = 'dummy_cat'
            self.grouping_name = 'category'
        else:
            self.grouping_name = col_names[grouping_idx]

        self.timestamp_idx = timestamp_idx
        self.count_data = count_data
        self.mask_value = mask_value

        if self.data is None:
            raise ValueError('Must provide a Pandas df to instantiate this class')

        # set name of target variable
        if target_idx is None:
            raise ValueError('Must provide an index for a target column to instantiate this class')
        self.data = self.data.rename(columns={col_names[target_idx]:'target'})

        # delete index column if one exists (absolute position information only available through covariates)
        if index_col is not None:
            self.data = self.data.drop(col_names[index_col], axis=1)

        # augment dataset with covariates
        time_name = self._sort_by_timestamp(col_names)
        self._age_augmentation()
        self._datetime_augmentation(time_name)

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

        # standardize
        self._standardize(val_split)

        # store number of features and categorical count and target means
        self.features = self.data.shape[1] - 1 # no 1) target or 2) grouping name, add 3) prev target
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
        self.data[time_name] = pd.to_datetime(self.data[time_name], unit = 's')
        return time_name

    def _age_augmentation(self):
        """
        util function
            augment dataset with age covariate
        """
        # age (timesteps from 0 for each unique time series)
        self.data['age'] = self.data.groupby(self.grouping_name).cumcount()
        self.train_set_ages = self.data.groupby(self.grouping_name)['age'].agg('max')
        self.train_set_ages['dummy_test_category'] = 0

    def _datetime_augmentation(self, time_name):
        """
        util function
            augment dataset with datetime covariates
        """
         # datetime features
        self.data['hour_of_day'] = self.data[time_name].dt.hour
        self.data['day_of_week'] = self.data[time_name].dt.dayofweek
        self.data['day_of_month'] = self.data[time_name].dt.day
        self.data['day_of_year'] = self.data[time_name].dt.dayofyear
        self.data['week_of_year'] = self.data[time_name].dt.weekofyear
        self.data['month_of_year'] = self.data[time_name].dt.month
        self.data['year'] = self.data[time_name].dt.year
        self.data = self.data.drop(time_name, axis=1)

    def _create_sampling_dist(self):
        """
        util function
            create scaled sampling distribution over time series 
        """

        self.scale_factors = 1 + self.target_means
        #self.scale_factors = self.scale_factors.apply(lambda x: max(0, x))

        # softmax the distribution for sampling
        e_x = np.exp(self.scale_factors - np.max(self.scale_factors))
        self.scale_factors_softmax = e_x / e_x.sum(axis = 0)
        
    def _train_val_split(self, val_split):
        """
        util function
            split dataset object into training and validation data frames
        """
        # split data into training and validation sets
        assert val_split >= 0 and val_split < 1, \
            'Validation split must be between 0 (inclusive) and 1 (exclusive)'

        if val_split == 0:
            self.train_data = self.data
        else:
            nrow = int(self.data.shape[0] * val_split)
            self.train_data = self.data.head(self.data.shape[0] - nrow)
            self.val_data = self.data.tail(nrow)

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
        for idx in pd.isnull(df)['target'].nonzero()[0]:
            key = df[self.grouping_name][idx]
            if key in self.missing_tgt_vals.keys():
                self.missing_tgt_vals[key].append(df['age'][idx])
            else:
                self.missing_tgt_vals[key] = [df['age'][idx]]

    def _standardize(self, val_split):
        """ 
        util function
            standardize covariates and record locations of missing tgt values (for standardization later)
        """
        # standardize covariates N(0,1) and 'target' col by mean
        covariate_mask = [False if col_name == 'target' or col_name == self.grouping_name
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

    def _one_hot_padding(self, pandas_df, padding_df):
        """
        Util padding function
            :param padding_df:
            :param one_hot_root_list:
            :return: padding_df

        from https://github.com/arrigonialberto86/deepar
        """
        for one_hot_root in self.one_hot_indices:
            one_hot_columns = [i for i in pandas_df.columns   # select columns equal to 1
                               if i.startswith(one_hot_root) and pandas_df[i].values[0] == 1]
            for col in one_hot_columns:
                padding_df[col] = 1
        return padding_df

    def _pad_ts(self, pandas_df, desired_len, padding_val=0):
        """
        Add padding int to the time series
            :param pandas_df:
            :param desired_len: (int)
            :param padding_val: (int)
            :return: X (feature_space), y
        
        from https://github.com/arrigonialberto86/deepar
        """
        pad_length = desired_len - pandas_df.shape[0]
        padding_df = pd.DataFrame({col: padding_val for col in pandas_df.columns}, 
            index=[i for i in range(pad_length)])

        if self.one_hot_indices:
            padding_df = self._one_hot_padding(pandas_df, padding_df)

        return pd.concat([padding_df, pandas_df]).reset_index(drop=True)

    def _sample_ts(self, pandas_df, desired_len, padding_val = 0, negative_obs = 1):
        """
            :param pandas_df: input pandas df with 'target' columns e features
            :param desired_len: desired sample length (number of rows)
            :param padding_val: default is 0
            :param negative_obs: how far before beginning of time series is it possible to set t
            :return: a pandas df (sample)
        
        from https://github.com/arrigonialberto86/deepar
        """
        if pandas_df.shape[0] < desired_len:
            raise ValueError('Desired sample length is greater than df row len')
        if pandas_df.shape[0] == desired_len:
            return pandas_df

        start_index = np.random.choice([i for i in range(0 - negative_obs, pandas_df.shape[0] - desired_len)])

        # replace beginning of series with padded values to learn beginning
        if start_index < 0:
            return self._pad_ts(pandas_df.head(desired_len + start_index), desired_len, padding_val = padding_val)

        return pandas_df.iloc[start_index: start_index+desired_len, ]

    def _add_prev_target_col(self, df):
        """
        util function
            add column with previous target value for autoregressive modeling
        """
        # add feature column for previous output value (z_{t-1})
        df = df.reset_index(drop=True)
        df.loc[:,'prev_target'] = pd.Series([0]).append(df['target'].iloc[:-1], ignore_index=True)

        # scale
        df.loc[:, 'prev_target'] = \
            df['prev_target'] / self.target_means[df[self.grouping_name]].reset_index(drop = True)

        # interpolate (will only replace NA rows (first time))
        df.loc[:,'prev_target'] = df['prev_target'].interpolate()

        # replace target missing rows with mask
        df.loc[:, 'target'] = df['target'].fillna(self.mask_value)

        return df

    def _sample_missing_tgts(self, df, model, category, missing_tgt_vals, window_size, batch_size, include_target = True):
        """
        util function
            sample missing target values from current model parameters
        """

        # sample missing 'targets' from current model parameters (for 'prev_targets')
        if category in missing_tgt_vals.keys():
            if not set(missing_tgt_vals[category]).isdisjoint(df['age'].iloc[:-1]):
                drop_list = [self.grouping_name]
                if include_target:
                    drop_list.append('target')
                continuous = df.drop(drop_list, 1).values.reshape(1, window_size, -1)
                continuous = np.repeat(continuous, batch_size, axis = 0)
                categorical = df[self.grouping_name].values.reshape(1, window_size)
                categorical = np.repeat(categorical, batch_size, axis = 0)
                preds = model([continuous, categorical], training = True)[0][0]

                # refill indices 
                refill_indices = df.index[df['age'].isin(missing_tgt_vals[category])]
                refill_values = [preds[i] for i in [r - df.index[0] for r in refill_indices]]
                for idx, val in zip(refill_indices, refill_values):
                    df['prev_target'][idx] = val
        return df

    def next_batch(self, model, batch_size, window_size, verbose=False, padding_value=0, val_set = False):
        """
            :param model: model object, allows sampling for missing target obs. in training set
            :param batch_size: how many time series to be sampled in this batch (int)
            :param window_size: window of each sampled time series
            :param verbose: default false
            :param padding_value: default 0
            :param val_set: boolean, whether this generator should sample for training or validation
            :return: [X_continouous, X_categorical], C (categorical grouping variable), y

            bootstrapped from https://github.com/arrigonialberto86/deepar
        """

        # save padding value for test object
        self.padding_value = padding_value

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
            cat_data = self._add_prev_target_col(cat_data)

            # Initial padding for each selected time series to reach window_size
            if cat_data.shape[0] < window_size:
                sampled_cat_data = self._pad_ts(pandas_df=cat_data,
                                                desired_len=window_size,
                                                padding_val=padding_value)
            # sample window from time series
            else:
                sampled_cat_data = self._sample_ts(pandas_df=cat_data,
                                                    desired_len=window_size,
                                                    padding_val=padding_value,
                                                    negative_obs=self.negative_obs)

            # sample missing 'targets' from current model parameters (for 'prev_targets')
            sampled_cat_data = self._sample_missing_tgts(sampled_cat_data, model, cat, self.missing_tgt_vals, 
                window_size, batch_size)
   
            sampled.append(sampled_cat_data)
        
        data = pd.concat(sampled)

        # [cont_inputs, cat_inputs], cat_labels, targets
        return ([data.drop(['target', self.grouping_name], 1).values.reshape(batch_size, window_size, -1),
               data[self.grouping_name].values.reshape(batch_size, window_size)], 
               tf.constant(cat_samples.reshape(batch_size, 1), dtype = tf.int32),
               tf.constant(data['target'].values.reshape(batch_size, window_size, 1), dtype = tf.float32))

class TimeSeriesTest(TimeSeries):

    def __init__(self, pandas_df, train_ts_obj, target_idx = None, timestamp_idx = None, 
            grouping_idx=None, one_hot_indices=None, index_col = None, mask_value = -1):
        """
            :param pandas_df: df of test time series
            :param train_ts_obj: TimeSeries object defined on training / validation set
            :param target_idx: index of target column, if None Error will be raised
            :param timestamp_idx: index of column containing timestamps, timestamps are parsed as # of seconds
            :param grouping_idx: index of grouping column
            :param one_hot_indices: list of indices of columns that are one hot encoded
            :param index_col: index column, if not None will be dropped
            :param mask_value: mask to use on testing timestep indices > 1 (bc predictions batched)
        """
        
        self.data = pandas_df
        self.one_hot_indices = one_hot_indices
        col_names = list(self.data)
        if grouping_idx is None:
            self.data['category'] = 'dummy_cat'
            self.grouping_name = 'category'
        else:
            self.grouping_name = col_names[grouping_idx]
        self.timestamp_idx = timestamp_idx
        self.count_data = train_ts_obj.count_data
        self.train_ts_obj = train_ts_obj
        self.mask_value = mask_value

        if self.data is None:
            raise ValueError('Must provide a Pandas df to instantiate this class')

        # delete target column if one exists (not needed for test)
        if target_idx is not None:
            self.data = self.data.drop(col_names[target_idx], axis=1)

        # delete index column if one exists (absolute position information only available through covariates)
        if index_col is not None:
            self.data = self.data.drop(col_names[index_col], axis=1)

        # sort df by timestamp
        time_name = self._sort_by_timestamp(col_names)

        # age (timesteps from beginning of train set for each unique time series)
        self.test_groups = self.data[self.grouping_name].unique()
        self.new_test_groups = []
        for group in self.test_groups:
            if group not in train_ts_obj.unique_cats:
                train_ts_obj.train_set_ages[group] = 0
                self.new_test_groups.append(group)
        self.data['age'] = self.data.groupby(self.grouping_name).cumcount()
        self.data['age'] += train_ts_obj.train_set_ages[self.data[self.grouping_name]].reset_index(drop=True)

         # datetime features
        self._datetime_augmentation(time_name)

        # standardize covariates N(0,1)
        covariate_mask = [False if col_name == self.grouping_name else True for col_name in self.data.columns]
        self.data.loc[:, covariate_mask] = train_ts_obj.scaler.transform(self.data.loc[:, covariate_mask].astype('float'))

        # compute max prediction horizon
        self.horizon = self.data.groupby(self.grouping_name)['age'].count().max()

        # assert compatibility with training TimeSeries object)
        assert self.data.shape[1] == train_ts_obj.features, \
            "Number of feature columns in test object must be equal to the number in train object"
        assert self.count_data == train_ts_obj.count_data, \
            "Count data boolean in test object must be equivalent to train object"

        self.prepared = False

    def _sort_by_timestamp(self, *args, **kwargs):
        return super(TimeSeriesTest, self)._sort_by_timestamp(*args, **kwargs)

    def _datetime_augmentation(self, *args, **kwargs):
        super(TimeSeriesTest, self)._datetime_augmentation(*args, **kwargs)

    def _pad_ts(self, *args, **kwargs):
        return super(TimeSeriesTest, self)._pad_ts(*args, **kwargs)

    def _add_prev_target_col(self, df, train_df = None):
        """
        util function
            add column with previous target value for autoregressive modeling
        """
        df = df.reset_index(drop=True)
        if train_df is None:

            # add feature column for previous output value (z_{t-1})
            df.loc[:,'prev_target'] = pd.Series([0]).append(df['target'].iloc[:-1], ignore_index=True)

            # scale
            df.loc[:, 'prev_target'] = \
                df['prev_target'] / self.train_ts_obj.target_means[df[self.grouping_name]].reset_index(drop = True)

            # interpolate (will only replace NA rows (first time))
            df.loc[:,'prev_target'] = df['prev_target'].interpolate()

        else:
            df.loc[:, 'prev_target'] = \
                train_df['target'].tail(1).repeat(repeats = df.shape[0]).reset_index(drop = True)
            df.loc[:, 'prev_target'] = \
                df['prev_target'] / self.train_ts_obj.target_means[df[self.grouping_name]].reset_index(drop = True)

        return df

    def _sample_missing_tgts(self, *args, **kwargs):
        return super(TimeSeriesTest, self)._sample_missing_tgts(*args, **kwargs)

    def _prepare_batched_test_data(self, batch_size, window_size, include_all_training = False, verbose = False):
        """
        Split data into batches of window_size (all batches include all categories) for stateful inference
            :param batch_size: batch size
            :param window_size: window of each sampled time series
            :param include_all_training: whether to include all training data in prep of batches
            :param verbose: default false
        """

        if include_all_training:
            max_train_age = self.train_ts_obj.data.groupby(self.train_ts_obj.grouping_name)['target'].count().max()
            if max_train_age % window_size != 0:
                max_train_age = (max_train_age // window_size) * window_size + window_size
        else:
            max_train_age = window_size

        # calculate # train and test batches
        self.train_batch_ct = max_train_age // window_size
        # last train batch produces first test output
        self.test_batch_ct = self.train_batch_ct + self.horizon

        data = []
        self.scale_keys = []
        for cat in self.test_groups:
            if cat in self.new_test_groups:
                train_data = pd.DataFrame({col: self.train_ts_obj.padding_value for col in self.data.columns}, 
                    index=[i for i in range(max_train_age)])
                
                # add 'prev_target' column for this series
                train_data = self._add_prev_target_col(train_data)
            
            else:
                enc_cat = self.train_ts_obj.label_encoder.transform([cat])[0]

                train_data = self.train_ts_obj.data[self.train_ts_obj.data[self.grouping_name] == enc_cat]

                # add 'prev_target' column for this series
                train_data = self._add_prev_target_col(train_data)
                if train_data.shape[0] < max_train_age:
                    train_data = self._pad_ts(pandas_df=train_data,
                                                    desired_len=max_train_age,
                                                    padding_val=self.train_ts_obj.padding_value)
                else:
                    train_data = train_data.tail(max_train_age)
                    train_data = train_data.reset_index(drop=True)

            # convert groups to ints in test data
            test_data = self.data[self.data[self.grouping_name] == cat]
            if cat in self.new_test_groups:
                test_data[self.grouping_name] = 'dummy_test_category'
            test_data[self.grouping_name] = self.train_ts_obj.label_encoder.transform(test_data[self.grouping_name])
            
            # add prev target w/ same value for all rows to test and scale
            test_data = self._add_prev_target_col(test_data, train_df = train_data)

            # append test data, drop 'target' col from training data
            prepped_data = \
                pd.concat([train_data.drop('target', axis=1), test_data]).reset_index(drop = True)
            data.append(prepped_data)
            
            # track scale keys for inverse scaling at inference
            self.scale_keys.append(prepped_data[self.grouping_name][0])

        self.prepped_data = data
        self.prepared = True
        self.batch_idx = 0

        # pad scale keys until batch size
        self.scale_keys.extend([self.scale_keys[0]] * (batch_size - len(self.scale_keys)))
        self.scale_keys = \
            tf.constant(np.array(self.scale_keys).reshape(batch_size, 1), dtype = tf.int32)

    def _one_hot_padding(self, *args, **kwargs):
        return super(TimeSeriesTest, self)._one_hot_padding(*args, **kwargs)

    def next_batch(self, model, batch_size, window_size, include_all_training = False, verbose=False):
        """
        Split data into batches of window_size (all batches include all categories) for stateful inference
            :param batch_size: batch size
            :param window_size: window of each sampled time series
            :param include_all_training: whether to include all training data in prep of batches
            :param verbose: default false
            :return [X_continouous, X_categorical], C (categorical grouping variable), prediction_horizon_index
        """

        if not self.prepared:
            self._prepare_batched_test_data(batch_size, 
                window_size, 
                include_all_training = include_all_training, 
                verbose = verbose)

        # return -1 if no more batches
        if self.batch_idx == self.test_batch_ct:
            return (None, None)
        
        # grab current batch
        if self.batch_idx >= self.train_batch_ct:
            batch_data = []
            start_idx = self.train_batch_ct * window_size + self.batch_idx - self.train_batch_ct
            for df in self.prepped_data:
                # mask continuous inputs so timesteps > 0 will be skipped in lstm inference
                datum = df.iloc[start_idx:start_idx + window_size, :].reset_index(drop = True)
                datum.loc[1:, datum.columns != self.grouping_name] = self.mask_value
                if window_size - datum.shape[0] > 0:
                    datum = datum.append(pd.concat([datum.iloc[-1:, :]] * (window_size - datum.shape[0]), 
                        ignore_index = True))
                batch_data.append(datum)
        else:
            batch_data = [df.iloc[self.batch_idx * window_size:(self.batch_idx + 1) * window_size, :] 
                for df in self.prepped_data] 
                
            # sample missing 'targets' from current model parameters (for 'prev_targets')
            batch_data = [self._sample_missing_tgts(b_data, model, b_data[self.grouping_name].iloc[0], 
                self.train_ts_obj.missing_tgt_vals, window_size, batch_size, include_target = False) 
                for b_data in batch_data]

        batch_data = pd.concat(batch_data)
        self.batch_idx += 1

        # 'prev_target' in test batches will be overwritten during ancestral sampling in predict, 
        # so doesn't matter if dropped here
        x_cont = batch_data.drop([self.grouping_name], 1).values.reshape(len(self.test_groups), window_size, -1)
        x_cat = batch_data[self.grouping_name].values.reshape(len(self.test_groups), window_size)
        x_cont = np.append(x_cont, [x_cont[0]] * (batch_size - len(self.test_groups)), axis = 0)
        x_cat = np.append(x_cat, [x_cat[0]] * (batch_size - len(self.test_groups)), axis = 0)
        return ([x_cont, x_cat], self.batch_idx - self.train_batch_ct) 

def train_ts_generator(model, ts_obj, batch_size, window_size, verbose = False, padding_value = 0, val_set = False):
    """
    This is a util generator function
        :param model: model (with current parameters) to sample missing tgt values
        :param ts_obj: a TimeSeries class object that implements the 'next_batch' method
        :param batch_size: batch size
        :param window_size: window of each sampled time series
        :param verbose: default false
        :param padding_value: default 0
        :param val_set: boolean for mode, True is validation 
        :yield: [X_continouous, X_categorical], C (categorical grouping variable), y

    bootstrapped from https://github.com/arrigonialberto86/deepar
    """
    while 1:
        yield ts_obj.next_batch(model, 
                batch_size, 
                window_size, 
                verbose = verbose, 
                padding_value = padding_value, 
                val_set = val_set)
    
def test_ts_generator(model, ts_obj, batch_size, window_size, include_all_training = False, verbose = False):
    """
    This is a util generator function
        :param model: model (with current parameters) to sample missing tgt values
        :param ts_obj: a TimeSeriesTest class object that implements the 'next_batch' method
        :param batch_size: batch size
        :param window_size: window of each sampled time series
        :param include_all_training: whether to start inference at beginning of training set
        :param verbose: default false
        :yield: [X_continouous, X_categorical], C (categorical grouping variable), prediction_horizon_index
    """
    
    while 1:
        x_test, horizon_idx = ts_obj.next_batch(model, 
            batch_size, 
            window_size, 
            include_all_training = include_all_training,
            verbose = verbose)
        x_test, horizon_idx
        yield x_test, horizon_idx


