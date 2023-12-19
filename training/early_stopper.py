from abc import abstractmethod
import numpy as np


def make_early_stopper(*args, **kwargs):
    return EarlyStoppingEnsemble(*args, **kwargs)


class EarlyStoppingBase:
    def __init__(self):
        self.message = None

    @abstractmethod
    def stop(self, *args, **kwargs):
        pass


class EarlyStoppingEnsemble(EarlyStoppingBase):
    def __init__(self, *early_stoppers, **kwargs):
        super().__init__()
        self.early_stoppers = early_stoppers
        self.patience = kwargs.get('patience', 1)
        self.stop_count = 0

    def stop(self, *args, **kwargs):
        crit = any([early_stopper.stop(*args, **kwargs) for early_stopper in self.early_stoppers])
        self.stop_count += 1 if crit else 0
        self.message = [early_stopper.message for early_stopper in self.early_stoppers if early_stopper.message is not None]
        return self.stop_count >= self.patience


class EarlyStoppingDiscLoss(EarlyStoppingBase):
    def __init__(self, loss_threshold, check_last=1):
        super().__init__()
        self.loss_threshold = loss_threshold
        self.check_last = check_last

    def stop(self, losses):
        crit = np.abs(np.mean(losses['D'][-1*self.check_last:])) < self.loss_threshold
        if crit:
            self.message = 'Early stopping: Discriminator loss below threshold'
        return crit
