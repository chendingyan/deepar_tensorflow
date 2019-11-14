import fire
import os
from deepar.dataset.time_series import TimeSeries, TimeSeriesTest
import pandas as pd
import numpy as np
from deepar.model.learner import DeepARLearner
from sklearn.metrics import mean_squared_error
from math import sqrt


def _time_col_to_seconds(df, dataset):
    if dataset == '56_sunspots':
        df['year'] = pd.to_timedelta(pd.to_datetime(df['year'], format='%Y')).dt.total_seconds()
    elif dataset == 'LL1_736_population_spawn':
        df['day'] = pd.to_timedelta(pd.to_datetime(df['day'], unit='D')).dt.total_seconds()
    return df

def _multi_index_prep(df, dataset):
    if dataset == 'LL1_736_population_spawn':
        df = df.set_index(['species', 'sector'])
        df['group'] = df.index
        df['group'] = df['group'].apply(lambda x: " ".join(x))
        df = df.reset_index(drop=True)
    return df

def _create_ts_object(df, dataset):
    if dataset == '56_sunspots':
        ds = TimeSeries(df, target_idx = 4, timestamp_idx = 1, index_col=0)
    elif dataset == 'LL1_736_population_spawn':
        ds = TimeSeries(df, target_idx = 2, timestamp_idx = 1, index_col=0, grouping_idx = 3, count_data = True)
    return ds

def _create_ts_test_object(df, train_ds, dataset):
    if dataset == '56_sunspots':
        test_ds = TimeSeriesTest(df, train_ds, target_idx = 4, timestamp_idx = 1, index_col=0)
    elif dataset == 'LL1_736_population_spawn':
        test_ds = TimeSeriesTest(df, train_ds, target_idx = 2, timestamp_idx = 1, index_col=0, grouping_idx = 3)
    return test_ds

def train(working_dir, dataset = '56_sunspots', epochs = 1):
    # read in training df
    df = pd.read_csv(f'/data/datasets/seed_datasets_current/{dataset}/TRAIN/dataset_TRAIN/tables/learningData.csv')#.format(dataset))
    df = _multi_index_prep(df, dataset)
    df = _time_col_to_seconds(df, dataset)

    # create TimeSeries and Learner objects
    ds = _create_ts_object(df, dataset)
    learner = DeepARLearner(ds, verbose=1)

    # fit
    learner.fit(epochs = epochs, 
        batches = 16, 
        early_stopping = True, 
        checkpoint_dir = os.path.join("./checkpoints", working_dir))

    # evaluate
    test_df = pd.read_csv('/data/datasets/seed_datasets_current/{}/TEST/dataset_TEST/tables/learningData.csv'.format(dataset))
    test_df = _multi_index_prep(test_df, dataset)
    test_df = _time_col_to_seconds(test_df, dataset)
    test_ds = _create_ts_test_object(test_df, ds, dataset)

    preds = learner.predict(test_ds, 
        horizon = None, 
        samples = 1, 
        include_all_training = True).reshape(-1)

    # scores = pd.read_csv('../datasets/seed_datasets_current/56_sunspots/SCORE/dataset_SCORE/tables/learningData.csv')
    # rms = sqrt(mean_squared_error(scores['sunspots'], preds))
    
if __name__ == '__main__':
  fire.Fire(train)
