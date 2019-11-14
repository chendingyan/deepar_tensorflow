import tensorflow as tf

from tensorflow.keras.layers import Input, Dense, LSTM, Masking, Embedding, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.activations import relu, softplus
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsoluteError, Mean

from deepar.model.loss import unscale, GaussianLogLikelihood, NegativeBinomialLogLikelihood, build_tf_lookup
from deepar.dataset.time_series import train_ts_generator, test_ts_generator
from deepar.model.layers import LSTMResetStateful, GaussianLayer
from deepar.model.callbacks import EarlyStopping

import logging
import numpy as np
import os
import time
import sys

class DeepARLearner:
    def __init__(self, ts_obj, output_dim = 1, emb_dim = 128, lstm_dim = 128, dropout = 0.1, 
        optimizer = 'adam', lr = 0.001, batch_size = 16, scheduler = None, train_window = 20, verbose = 0, 
        logger = None, inference_mask = -1):
        """
            initialize DeepAR model
                :param ts_obj: DeepAR time series Dataset
                :param outpout_dim: dimension of output variables 
                :param emb_dim: dimension of categorical embeddings 
                :param lstm_dim: dimension of lstm cells
                :param dropout: default 0.1
                :param optimizer: default adam
                :param lr: default 0.001
                :param batch_size: number of time series to sample in each batch
                :param scheduler: default None (use adam)
                :param train_window: the length of time series sampled for training. consistent throughout
                :param verbose: default false
                :param logger: logger object
                :param inference_mask: mask to use on testing timestep indices > 1 (bc predictions batched)
        """

        self.ts_obj = ts_obj
        self.train_window = train_window
        if logger is None:
            logger = logging.getLogger('deepar')
            handler = logging.StreamHandler(sys.stdout)
            logger.addHandler(handler)
        self.logger = logger
        assert verbose == 0 or verbose == 1, 'argument verbose must be 0 or 1'
        if verbose == 1:
            self.logger.setLevel(logging.INFO)
        self.verbose = verbose
        self.scheduler = scheduler
        self.lr = lr

        # make sure batch_size is at least as big as # of groups in training set 
        # (to support batched inference over all groups during test)
        if batch_size < ts_obj.num_cats:
            batch_size = ts_obj.num_cats
        self.batch_size = batch_size
        
        # define model
        self.model = self._create_model(self.ts_obj.num_cats, 
            self.ts_obj.features, 
            output_dim=output_dim,
            emb_dim=emb_dim, 
            lstm_dim=lstm_dim,
            dropout=dropout,
            count_data=self.ts_obj.count_data,
            inference_mask=inference_mask)

        # define optimizer
        if optimizer == 'adam':
            self.optimizer = Adam(learning_rate = lr)
        elif optimizer == 'sgd':
            self.optimizer = SGD(lr = lr, momentum=0.1, nesterov=True)
        else:
            self.optimizer = optimizer

        # define loss function
        if self.ts_obj.count_data:
            self.loss_fn = NegativeBinomialLogLikelihood(self.ts_obj.scale_factors, 
                mask_value = self.ts_obj.mask_value)
        else:
            self.loss_fn = GaussianLogLikelihood(self.ts_obj.scale_factors,
                mask_value = self.ts_obj.mask_value)

    def _create_model(self, num_cats, num_features, output_dim = 1, emb_dim = 128, lstm_dim = 128, 
        batch_size = 16, dropout = 0.1, count_data = False, inference_mask = -1):
        """ 
        util function
            creates model architecture (Sequential) with arguments specified in constructor
        """

        cont_inputs = Input(shape = (self.train_window, num_features), batch_size = self.batch_size)
        cat_inputs = Input(shape = (self.train_window,), batch_size = self.batch_size)

        embedding = Embedding(num_cats, emb_dim)(cat_inputs)

        masked_input = Masking(mask_value = inference_mask)(cont_inputs)
        concatenate = Concatenate()([masked_input, embedding])

        lstm_out = LSTMResetStateful(lstm_dim, 
            return_sequences = True,
            stateful = True,
            dropout = dropout, 
            recurrent_dropout = dropout, 
            name = 'lstm')(concatenate)

        mu = Dense(output_dim, 
            kernel_initializer = 'glorot_normal',
            bias_initializer = 'glorot_normal',
            name = 'mu')(lstm_out)

        sigma = Dense(output_dim, 
            kernel_initializer = 'glorot_normal',
            bias_initializer = 'glorot_normal',
            name = 'sigma')(lstm_out)
        
        model = Model(inputs = [cont_inputs, cat_inputs], outputs = [mu, sigma])

        return model

    def _training_loop(self, train_gen, val_gen, epochs = 100, batches = 50, early_stopping = True, 
        stopping_patience = 0):
        """ 
        util function
            iterates over batches, updates gradients, records metrics, writes to tb, checkpoints, early stopping
        """

        # set up metrics to track during training
        batch_loss_avg = Mean()
        epoch_loss_avg = Mean()
        eval_loss_avg = Mean()
        eval_mae = MeanAbsoluteError()
        eval_rmse = RootMeanSquaredError()
        eval_mae_avg = Mean()
        eval_rmse_avg = Mean()

        # set up early stopping callback
        EarlyStopping(patience = stopping_patience, active = early_stopping)

        # setup table for unscaling
        lookup_table = build_tf_lookup(self.ts_obj.target_means)

        # Iterate over epochs.
        for epoch in range(epochs):
            self.logger.info(f'Start of epoch {epoch}')
            start_time = time.time()
            for batch, (x_batch_train, cat_labels, y_batch_train) in enumerate(train_gen):

                # compute loss
                with tf.GradientTape(persistent = True) as tape:
                    preds = self.model(x_batch_train, training = True) 
                    
                    # softplus parameters 
                    scale = softplus(preds[1])
                    if self.ts_obj.count_data:
                        mu = softplus(preds[0])

                    mu, scale = unscale(mu, scale, cat_labels, lookup_table)
                    loss_value = self.loss_fn(y_batch_train, preds)
                
                # sgd
                tf.summary.scalar('train_loss', loss_value, epoch * batches + batch)
                batch_loss_avg(loss_value)
                epoch_loss_avg(loss_value)
                grads = tape.gradient(loss_value, self.model.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

                # Log 5x per epoch.
                if batch % (batches // 5) == 0 and batch != 0:
                    self.logger.info(f'Epoch {epoch}: Avg train loss over last {(batches // 5)} batches: {batch_loss_avg.result()}')
                    batch_loss_avg.reset_states()
                
                # Run each epoch batches times
                if batch == batches:
                    self.logger.info(f'Epoch {epoch} took {round(time.time() - start_time, 0)}s : Avg train loss: {epoch_loss_avg.result()}')
                    break
                
            # validation
            if val_gen is not None:
                self.logger.info(f'End of epoch {epoch}, validating...')
                start_time = time.time()
                for batch, (x_batch_val, cat_labels, y_batch_val) in enumerate(val_gen):
                    
                    # compute loss, doesn't need to be persistent bc not updating weights
                    with tf.GradientTape() as tape:

                        # treat as training -> reset lstm states inbetween each batch
                        preds = self.model(x_batch_val, training = True) 

                        # softplus parameters 
                        scale = softplus(preds[1])
                        if self.ts_obj.count_data:
                            mu = softplus(preds[0])

                        mu, scale = unscale(mu, scale, cat_labels, lookup_table)
                        loss_value = self.loss_fn(y_batch_val, preds)

                    # log validation metrics (avg loss, avg MAE, avg RMSE)
                    eval_mae(y_batch_val, preds[0])
                    eval_rmse(y_batch_val, preds[0])
                    eval_loss_avg(loss_value)
                    if batch == batches:
                        break

                # logging
                self.logger.info(f'Validation took {round(time.time() - start_time, 0)}s')
                self.logger.info(f'Epoch {epoch}: Val loss on {batches} batches: {eval_loss_avg.result()}')
                self.logger.info(f'Epoch {epoch}: Val MAE: {eval_mae_avg.result()}, RMSE: {eval_rmse_avg.result()}')
                tf.summary.scalar('val_loss', eval_loss_avg.result(), epoch)
                tf.summary.scalar('val_mae', eval_mae_avg.result(), epoch)
                tf.summary.scalar('val_rmse', eval_rmse_avg.result(), epoch)

                # early stopping
                EarlyStopping(eval_loss_avg.result())

                # reset metric states
                eval_loss_avg.reset_states()
                eval_mae.reset_states()
                eval_rmse.reset_states()
            else:
                EarlyStopping(epoch_loss_avg.result())

            # reset epoch loss metric
            epoch_loss_avg.reset_states()

    def fit(self, checkpoint_dir = None, validation = True, batches=50, epochs=100, early_stopping = True,
        stopping_patience = 0):

        """ fits DeepAR model for batches * epochs iterations
                :param checkpoint_dir: directory to save checkpoint and tensorboard files
                :param validation: whether to perform validation. If True will automatically try
                    to use validation data sequestered in construction of time series object
                :param batches: number of batches to process each epoch
                :param epochs: number of epochs
                :param early_stopping: whether to include early stopping callback
                :param stopping_patience: early stopping callback patience, default 0
                :return:  
        """

        self.epochs = epochs

        # try to load previously saved checkpoint from filepath
        if checkpoint_dir is not None:
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            filepath = os.path.join(checkpoint_dir, "{epoch:04d}.ckpt")
            checkpointer = tf.train.Checkpoint(optimizer = self.optimizer, model = self.model)
            latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir)
            if latest_ckpt:
                checkpointer.restore(latest_ckpt)
        else:
            # create tensorboard log files in default location
            checkpoint_dir = "./tb/"
        tb_writer = tf.summary.create_file_writer(checkpoint_dir)
        tb_writer.set_as_default()

        # train generator
        train_gen = train_ts_generator(self.model, 
            self.ts_obj,
            self.batch_size,
            self.train_window, 
            verbose=self.verbose, 
            padding_value=0)

        # validation generator
        if validation:            
            val_gen = train_ts_generator(self.model, 
                self.ts_obj,
                self.batch_size, 
                self.train_window, 
                verbose=self.verbose, 
                padding_value=0, 
                val_set=True)
        else:
            val_gen = None

        # Iterate over epochs.
        self._training_loop(train_gen, 
            val_gen, 
            epochs=epochs,
            batches=batches,
            early_stopping=early_stopping,
            stopping_patience=stopping_patience)

    def predict(self, test_ts_obj, horizon = None, samples = 1, include_all_training = False):
        """ 
        predict horizon steps into the future
            :param test_ts_obj: time series object for prediction
            :param horzion: optional, can specify prediction horizon into future
            :param samples: how many samples to draw to calculate confidence interavls 
                (raw samples returned)
            :param include_all_training: whether to start calculating hidden states from beginning 
                    of training data, alternative is from t_0 - train_window
            :return: predictions [# unique test groups, horizon, # samples]
        """
        
        assert samples > 0, "The number of samples to draw must be positive"

        # test generator
        test_gen = test_ts_generator(self.model,
            test_ts_obj,
            self.batch_size,
            self.train_window, 
            include_all_training=include_all_training,
            verbose=self.verbose
        )

        # reset lstm states before prediction
        self.model.get_layer('lstm').reset_states()
        self.model.get_layer('lstm').reset_dropout_mask()
        self.model.get_layer('lstm').reset_recurrent_dropout_mask()

        # make sure horizon is legitimate value 
        # todo grouped horizon!
        if horizon is None or horizon > test_ts_obj.data.shape[0]:
            horizon = test_ts_obj.horizon
    
        # forward 
        test_samples = [[] for _ in range(len(test_ts_obj.test_groups))]
        start_time = time.time()
        for batch_idx, batch in enumerate(test_gen):
            
            x_test, horizon_idx = batch
            if horizon_idx is None:
                break
            if horizon_idx > horizon:
                break
            
            # don't need to replace for first test batch bc have tgt from last training example
            if horizon_idx > 1:
                # add one sample from previous predictions to test batches 
                # all dim 0, first row of dim 1, last col of dim 3
                x_test[0][:, :1, -1:] = mu

            # get first column predictions (squeeze 1st dim - horizon)
            mu, scale = self.model(x_test)
            mu, scale = mu[:, :1, :], scale[:, :1, :]

            # softplus parameters 
            scale = softplus(scale)
            if self.ts_obj.count_data:
                mu = softplus(mu)
            
            # draw samples from learned distributions for test samples
            if horizon_idx > 0:
                
                # unscale
                lookup_table = build_tf_lookup(self.ts_obj.target_means)
                scaled_mu, scaled_scale = unscale(mu, scale, test_ts_obj.scale_keys, lookup_table)

                # slice at number of unique ts and squeeze 1st dim - horizon, 2nd dim - output_dim)
                scaled_mu = tf.squeeze(scaled_mu[:len(test_ts_obj.test_groups)], [1, 2])
                scaled_scale = tf.squeeze(scaled_scale[:len(test_ts_obj.test_groups)], [1, 2])

                # sample from learned distribution
                # shape : [# test groups, samples]
                if self.ts_obj.count_data:
                    draws = [list(np.random.negative_binomial(mu, scale, samples)) for mu, scale in zip(scaled_mu, scaled_scale)]
                else:
                    draws = [list(np.random.normal(mu, scale, samples)) for mu, scale in zip(scaled_mu, scaled_scale)]

                # concatenate 
                for draw_list, sample_list in zip(draws, test_samples):
                    sample_list.append(draw_list)

        self.logger.info(f'Inference ({samples} sample(s), {horizon} timesteps) took {round(time.time() - start_time, 0)}s')
        
        # Shape [# test_groups, horizon, samples]
        # TODO [# test_groups, horizon, samples, output_dim] not supported yet
        return np.array(test_samples)
