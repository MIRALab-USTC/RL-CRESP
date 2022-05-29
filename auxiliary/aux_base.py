import abc
import torch


class AUXBase(object, metaclass=abc.ABCMeta):
    
    def __init__(self):
        self.network = None
        self.optimizer = None

    def train(self, training=True):
        self.network.train(training)

    def print_module(self):
        print("AuxiliaryTask:", self.network)

    @abc.abstractmethod
    def update_extr(data, num_aug, logger, step, save_log):
        pass

    def save(self, model_dir, step):
        torch.save(
            self.network.state_dict(), '%s/aux_net_%s.pt' % (model_dir, step)
        )
        self._save(model_dir, step)

    @abc.abstractmethod
    def _save(self, model_dir, step):
        pass

    def load(self, model_dir, step):
        self.network.load_state_dict(
            torch.load('%s/aux_net_%s.pt' % (model_dir, step))#, map_location=torch.device('cpu'))
        )
        self._load(model_dir, step)

    @abc.abstractmethod
    def _load(self, model_dir, step):
        pass

    @abc.abstractmethod
    def _print_log(self, logger):
        pass