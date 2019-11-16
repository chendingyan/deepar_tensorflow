import logging
logger = logging.getLogger(__name__)

class EarlyStopping(object):
    """ class that monitors a metric and stops training when the metric has stopped improving
    
        Args: 
            monitor_increase - if True, stops training when metric stops increasing. If False, 
                when metric stops decreasing. 
            patience - after how many epochs of degrading performance should training be stopped
            active - whether early stopping callback is active or not
    """

    def __init__(self, monitor_increase = False, patience = 0, active = True):

        self.monitor_increase = monitor_increase
        self.patience = patience
        self.prev_metric = None
        self.degrade_count = 0
        self.active = active

    def __call__(self, cur_metric):
        
        if not self.active:
            return False
        if self.prev_metric is None:
            self.prev_metric = cur_metric
            return False
        else:
            if self.monitor_increase:
                if cur_metric < self.prev_metric:
                    self.degrade_count += 1
                else:
                    self.degrade_count = 0
            else:
                if cur_metric > self.prev_metric:
                    self.degrade_count += 1
                else:
                    self.degrade_count = 0
            if self.degrade_count > self.patience:
                logging.info(f'Metric has degraded for {self.degrade_count} epochs, exiting training')
                return True
            self.prev_metric = cur_metric
            return False

    # def _callbacks(self, filepath, early_stopping = True, val_set = True, stopping_patience = 0, scheduler_factor = 0.2, 
    #         scheduler_patience = 5, min_cosine_lr = 0):
    #     """ defines callbacks that we want to use during fitting 
    #             :param filepath: filepath to save checkpoint and tensorboard files
    #             :param early_stopping: whether to include early stopping callback
    #             :param val_set: boolean, whether validation set exists
    #             :param stopping_patience: early stopping callback patience, default 0
    #             :param scheduler_factor: plateau lr scheduler decrease factor, default 0.2
    #             :param scheduler_patience: plateau lr scheduler patience, default 5
    #             :param min_cosine_lr: cosine annealing lr scheduler minimum lr, default 0
    #             :return: list of callbacks for fitting
    #     """

    #     if not val_set:
    #         monitor = 'loss'
    #     else:
    #         monitor = 'val_loss'

    #     callbacks = []

    #     # Model Checkpoint 
    #     checkpoint = ModelCheckpoint(filepath, 
    #         monitor=monitor, 
    #         verbose=self.verbose,
    #         save_best_only = True)

    #     # Tensorboard
    #     tb = TensorBoard(log_dir=filepath)

    #     callbacks = [checkpoint, tb]

    #     # Early Stopping
    #     if early_stopping:
    #         es = EarlyStopping(monitor=monitor, 
    #             patience = stopping_patience, 
    #             verbose = self.verbose)
    #         callbacks.append(es)

    #     # Learning Rate Scheduler (by default simply use Adam)
    #     if self.scheduler == 'plateau':
    #         scheduler = ReduceLROnPlateau(monitor=monitor, 
    #             factor = scheduler_factor, 
    #             patience=scheduler_patience, 
    #             verbose=self.verbose)
    #         callbacks.append(scheduler)

    #     elif self.scheduler == 'cosine_annealing':

    #         # custom cosine annealing schedule
    #         def cosine_annealing(epoch, cur_lr):
    #             return min_cosine_lr + (self.lr - min_cosine_lr) * (1 + math.cos(math.pi * epoch / self.epochs)) / 2

    #         scheduler = LearningRateScheduler(cosine_annealing, 
    #             verbose=self.verbose)
    #         callbacks.append(scheduler)
    #     return callbacks