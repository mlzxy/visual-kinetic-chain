import os
from modules.template import Mixin, Placeholder
import modules.data.kchain.sim as sim
import modules.data.augmentation as aug
import modules.data.kchain.base as base
from torch.utils.data import DataLoader
from typing import List

DEBUG = os.environ.get('DEBUG', '0') == '1'


class Config:
    batch_size: int = 16

    rlbench_path: str = "/scratch/xz653/datasets/fskeleton/RLBench" 
    calvin_path: str = "/scratch/xz653/datasets/fskeleton/Calvin"

    datasets: List[str] = ["rlbench"]
    num_workers: int = 1
    horizon: int = 10
    train_size: int = 10000
    eval_size: int = 1000

    multi_view = False

    stack_actions = True
    prob = None
    
    dataset_kwargs:dict = {}
     

class KChainDataLoader(Mixin):
    def __init__(self, cfg: Config):
        self.data_cfg: Config = cfg
    
    def get_dataset(self, limits, transform, cfg):
        datasets = []
        for d in cfg.datasets:
            if d == 'rlbench':
                d_kwargs = cfg.dataset_kwargs.get(d, {})
                d_path = getattr(cfg, d + '_path')
                D = sim.RLBenchDataset(d_path, transform=transform, horizon=cfg.horizon, multi_view=cfg.multi_view, **d_kwargs)
            elif d == 'calvin':
                d_kwargs = cfg.dataset_kwargs.get(d, {})
                d_path = getattr(cfg, d + '_path')
                D = sim.CalvinDataset(d_path, transform=transform,  horizon=cfg.horizon, multi_view=cfg.multi_view, **d_kwargs)
            else:
                raise KeyError()
            datasets.append(D)
        D = base.MultiDataset(datasets, limits=limits, probability=getattr(cfg, 'prob', None))
        return D

    @property
    def train_dataloader(self):
        transform = aug.image_aug_recipe(target_size=224, need_skeleton=True, normalize=True, **getattr(self.data_cfg, 'aug_kwargs', {}))
        D = self.get_dataset(self.data_cfg.train_size, transform, self.data_cfg)
        return DataLoader(D, batch_size=self.data_cfg.batch_size, shuffle=True, 
                        num_workers=self.data_cfg.num_workers, collate_fn=base.VLCollator(stack_actions=self.data_cfg.stack_actions,
                                                                                        stack_skels=getattr(self.data_cfg, 'stack_skels', True)), pin_memory=True)

    @property
    def eval_dataloader(self):
        if getattr(self.data_cfg, 'eval', None) is None:
            return None
        else:
            transform = aug.image_aug_recipe(target_size=224, need_skeleton=True, normalize=True, 
                                            color=0, spatial=0, hflip=0, vflip=0)
            D = self.get_dataset(self.data_cfg.eval.size, transform, self.data_cfg.eval)
            return DataLoader(D, batch_size=self.data_cfg.eval.batch_size, shuffle=False, 
                            num_workers=0, collate_fn=base.VLCollator(stack_actions=self.data_cfg.stack_actions,
                                                                     stack_skels=getattr(self.data_cfg, 'stack_skels', True)),
                            pin_memory=True)


if __name__ == "__main__":
    Config.eval_steps = 10
    class test(KChainDataLoader, Placeholder): pass
    DL = test(Config)
    for a in DL.eval_dataloader:
        print(a)
    