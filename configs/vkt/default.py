from typing import List, Dict
import socket
import os

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))

batch_size: int = 48 # 16
train_steps: int = 100000 * WORLD_SIZE
horizon: int = 20

DATASET_ROOT = '/filer/tmp1/xz653/datasets/fskeleton' # CHANGE THIS

class Config:
    class train:
        module: str = "modules.train.vkt_policy:LANG_SKEL_POLICY"
        save_every: int = 5000 # 5000
        eval_every: int = 2500 # 5000
        log_every: int = 20 # 20
        grad_clip: float = -1
        grad_clip_start: int = 40

    class module:
        class data:
            batch_size: int = batch_size

            rlbench_path: str = f"{DATASET_ROOT}/RLBench_slim"
            calvin_path: str = f"{DATASET_ROOT}/Calvin"

            datasets: List[str] = ["rlbench", "calvin"] #"rlbench", "calvin"
            prob = [0.7, 0.3]
            num_workers: int = 20
            horizon: int = horizon
            train_size: int = train_steps * batch_size
            eval_size: int = 4 * batch_size
            stack_actions = False
            need_heatmap = False
            multi_view = True
            
            dataset_kwargs: Dict = {
                'rlbench': {
                    'robot_p':  {'panda': 1.0, 'sawyer': 0., 'ur5': 0.} 
                },
                'calvin': {}
            }
            aug_kwargs = dict(color=0.1, spatial=0.0, hflip=0.5, vflip=0.0)

        class model:
            n_points: int = 200
            horizon = horizon
            hidden_size = 512
            depth = 8

            lora_modules: List[str] = ["q_proj", "v_proj", "q_proj", "out_proj", "c_proj", "ln_1", "ln_2", "c_fc", "ln_post", "ln_pre", "conv1"]
            # ["query", "value", "key", "dense", "projection"]
            lora_kwargs: Dict = {}

        class learn:
            lr: float = 0.0003
            weight_decay: float = 0.001
            steps: int = train_steps
            warmup_steps: int = 1000
    