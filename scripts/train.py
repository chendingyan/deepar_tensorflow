import fire
import os
from deepar.dataset.time_series import TimeSeriesTrain, TimeSeriesTest
import pandas as pd
import numpy as np
from deepar.model.learner import DeepARLearner
from sklearn.metrics import mean_squared_error
from math import sqrt
import sys
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
import logging
import time

logger = logging.getLogger('deepar')
handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


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
        ds = TimeSeriesTrain(df, target_idx=4, timestamp_idx=1, index_col=0)
    elif dataset == 'LL1_736_population_spawn':
        ds = TimeSeriesTrain(df, target_idx=2, timestamp_idx=1, index_col=0, grouping_idx=3, count_data=True)
    return ds


def _create_ts_test_object(df, train_ds, dataset):
    if dataset == '56_sunspots':
        test_ds = TimeSeriesTest(df, train_ds, target_idx=4, timestamp_idx=1, index_col=0)
    elif dataset == 'LL1_736_population_spawn':
        test_ds = TimeSeriesTest(df, train_ds, target_idx=2, timestamp_idx=1, index_col=0, grouping_idx=3)
    return test_ds


def train(working_dir, dataset='56_sunspots', epochs=100, stopping_patience=3):
    # read in training df
    df = pd.read_csv(f'../../datasets/seed_datasets_current/{dataset}/TRAIN/dataset_TRAIN/tables/learningData.csv')
    df = _multi_index_prep(df, dataset)
    df = _time_col_to_seconds(df, dataset)

    # create TimeSeries and Learner objects
    ds = _create_ts_object(df, dataset)
    learner = DeepARLearner(ds, verbose=1)

    # fit
    learner.fit(epochs=epochs,
                stopping_patience=stopping_patience,
                early_stopping=True,
                checkpoint_dir=os.path.join("./checkpoints", working_dir))

    # evaluate
    test_df = pd.read_csv(f'../../datasets/seed_datasets_current/{dataset}/TEST/dataset_TEST/tables/learningData.csv')
    test_df = _multi_index_prep(test_df, dataset)
    test_df = _time_col_to_seconds(test_df, dataset)
    test_ds = _create_ts_test_object(test_df, ds, dataset)

    preds = learner.predict(test_ds,
                            horizon=None,
                            samples=1,
                            include_all_training=True).reshape(-1)

    # scores = pd.read_csv('../datasets/seed_datasets_current/56_sunspots/SCORE/dataset_SCORE/tables/learningData.csv')
    # rms = sqrt(mean_squared_error(scores['sunspots'], preds))


def hp_search(working_dir, dataset='56_sunspots', epochs=100, metric='eval_mae_result', stopping_patience=5,
              stopping_delta=1):
    working_dir = os.path.join("./checkpoints", working_dir)

    # define domains for HP search
    HP_EMB_DIM = hp.HParam('emb_dim', hp.Discrete([32, 64, 128]))
    HP_LSTM_DIM = hp.HParam('lstm_dim', hp.Discrete([32, 64, 128]))
    HP_DROPOUT = hp.HParam('lstm_dropout', hp.Discrete([0.1, 0.2, 0.3]))
    HP_LR = hp.HParam('learning_rate', hp.Discrete([.0001, .001, .01]))
    HP_BS = hp.HParam('batch_size', hp.Discrete([32, 64, 128]))
    HP_WINDOW = hp.HParam('window_size', hp.Discrete([20, 40, 60]))

    # set up config 
    with tf.summary.create_file_writer(working_dir).as_default():
        hp.hparams_config(
            hparams=[HP_EMB_DIM, HP_LSTM_DIM, HP_DROPOUT, HP_LR, HP_BS, HP_WINDOW],
            metrics=[hp.Metric(metric, display_name=metric)]
        )

    # read in training df
    df = pd.read_csv(f'../../datasets/seed_datasets_current/{dataset}/TRAIN/dataset_TRAIN/tables/learningData.csv')
    df = _multi_index_prep(df, dataset)
    df = _time_col_to_seconds(df, dataset)

    # create TimeSeries and Learner objects
    ds = _create_ts_object(df, dataset)

    # grid search over parameters
    run_num = 0
    total_run_count = len(HP_EMB_DIM.domain.values) * \
                      len(HP_LSTM_DIM.domain.values) * \
                      len(HP_DROPOUT.domain.values) * \
                      len(HP_LR.domain.values) * \
                      len(HP_BS.domain.values) * \
                      len(HP_WINDOW.domain.values)

    # outfile for saving hp config and runtimes
    outfile = open(os.path.join(working_dir, "metrics.txt"), "w+", buffering=1)

    for emb_dim in HP_EMB_DIM.domain.values:
        for lstm_dim in HP_LSTM_DIM.domain.values:
            for dropout in HP_DROPOUT.domain.values:
                for lr in HP_LR.domain.values:
                    for bs in HP_BS.domain.values:
                        for window_size in HP_WINDOW.domain.values:
                            # create dict of parameters
                            hp_dict = {
                                'emb_dim': emb_dim,
                                'lstm_dim': lstm_dim,
                                'lstm_dropout': dropout,
                                'learning_rate': lr,
                                'batch_size': bs,
                                'window_size': window_size,
                            }
                            run_name = f'run-{run_num}'
                            logger.info(f'--- Starting Run: {run_name} of {total_run_count} ---')
                            # print_dict = {
                            #     h.name: hp_dict[h] for h in hp_dict
                            # }
                            logger.info(f'HP Dict: {hp_dict}')
                            hp.hparams(hp_dict)

                            # create learner and fit with these HPs
                            start_time = time.time()
                            learner = DeepARLearner(ds, verbose=1, hparams=hp_dict)
                            final_metric = learner.fit(epochs=epochs,
                                                       stopping_patience=stopping_patience,
                                                       stopping_delta=stopping_delta,
                                                       checkpoint_dir=os.path.join(working_dir, run_name))
                            outfile.write(
                                f'HPs: {hp_dict} ---- Metric: {final_metric} ---- Time: {round(time.time() - start_time, 2)}\n')
                            tf.summary.scalar(metric, final_metric, step=1)

                            run_num += 1
    outfile.close()


if __name__ == '__main__':
    fire.Fire({
        'train': train,
        'search': hp_search
    })
