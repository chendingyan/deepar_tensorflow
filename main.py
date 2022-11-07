from deepar.dataset.time_series import TimeSeriesTrain, TimeSeriesTest
from deepar.model.learner import DeepARLearner
import numpy as np
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
import os
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
        exit(-1)

import pandas as pd
train = pd.read_csv("data/train.csv", parse_dates = ["datetime"])
test = pd.read_csv("data/test.csv", parse_dates = ["datetime"])

train = train.drop(columns=['casual', 'registered'])

train_df = train[:10880]
val_df = train[10880:]

train_ds = TimeSeriesTrain(train_df, target_idx = 9, timestamp_idx = 0,freq='H', count_data=False)

learner = DeepARLearner(train_ds, verbose=1)


best_metric, epoch = learner.fit(epochs=10,
                stopping_patience=3,
                early_stopping=True,
                checkpoint_dir=os.path.join("./checkpoints2"))


test_ds = TimeSeriesTest(train_ds ,val_df,  target_idx = 9)


preds = learner.predict(test_ds, horizon = None, samples = 100, include_all_training = False, return_in_sample_predictions=False)
print(preds)