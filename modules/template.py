from abc import ABC, abstractmethod


class Mixin(ABC):
    
    @property
    @abstractmethod
    def train_dataloader(self):
        ...
    
    @property
    def eval_dataloader(self):
        pass
    
    @property
    @abstractmethod
    def model(self):
        ...
    
    @property
    def visualize(self):
        return None
         
    @property
    @abstractmethod
    def loss_function(self):
        """ return a loss_dict with at least a key `total`"""
        ...
        

class Placeholder(Mixin):

    @property
    def train_dataloader(self):
        ...
    
    @property
    def eval_dataloader(self):
        pass
    
    @property
    def model(self):
        ...

         
    @property
    def loss_function(self):
        """ return a loss_dict with at least a key `total`"""
        ...
    
    @property
    def visualize(self):
        ...

        
def Instantiate(cls, cfg, keys=['train_dataloader', 'eval_dataloader', 'model', 'loss_function', 'visualize']):
    inst = cls(cfg) 
    return [getattr(inst, k) for k in keys]


