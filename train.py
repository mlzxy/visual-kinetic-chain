import fire
import torch
import importlib
import yaml
import json
from tqdm import tqdm, trange
from accelerate import Accelerator
from require import require, get_exp_id, write_cfg
from modules.template import Instantiate
from pathlib import Path
from PIL import Image
import wandb as wandb_lib
import common
from time import time

class Config:
    class train:
        module: str = ""
        save_every: int = 5000
        eval_every: int = 5000
        log_every: int = 20
        grad_clip: int = -1
        grad_clip_start: int = 1000

    class module:
        pass 


def main(config, name=None, work_dir='./work_dir', wandb='no', save_model_by_step=False, 
        resume='', skip_first=0,
        vis_limit=60, pass_loss_to_model=False, **kwargs):
    config_text = Path(config).read_text()
    cfg: Config = require(config, toplevel=True, name=name, auto_name_with_parent=True)
    cfg = common.assign_to_class(cfg, kwargs)
    cfg = common.add_default(cfg, Config)
    cfg_dict = common.class_to_dict(cfg)
    use_wandb = wandb != 'no'
    
    work_dir = Path(work_dir) / get_exp_id()
    work_dir.mkdir(parents=True, exist_ok=True)
    write_cfg(work_dir / 'cfg.json', cfg_dict)
    (work_dir / 'config.py').write_text(config_text)
    accelerator = Accelerator(log_with='wandb' if use_wandb else None)
    if accelerator.is_main_process:
        logfile = open(work_dir / 'log.txt', "w")

    def init_trackers():
        if use_wandb: accelerator.init_trackers(wandb, config=cfg_dict, init_kwargs={'wandb': {'name': get_exp_id()}})
    def end_trackers():
        if use_wandb: accelerator.end_training()
    def lprint(msg, printer=print):
        if accelerator.is_main_process:
            print(msg, file=logfile, flush=False)
            printer(str(msg))
    def log_metrics(**kwargs):
        if use_wandb: accelerator.log(kwargs) 
    def log_figs(figs):
        if use_wandb:
            if isinstance(figs, dict):
                accelerator.log(figs)
            else:
                payload = dict(skeletons=[], predictions=[])
                for f in figs:
                    if isinstance(f, (str, Path)):payload['predictions'].append(wandb_lib.Video(str(f)))
                    elif isinstance(f, Image.Image): payload['skeletons'].append(wandb_lib.Image(f))
                    else: payload['predictions'].append(f)
                accelerator.log({k:v for k,v in payload.items() if len(v) > 0})
    to_item = lambda v: v.item() if hasattr(v, 'item') else v
         
    def dict_to_msg(dct):
        msg = f"[step:{str(bi).zfill(8)} time:{time()-start_time:.01f}s] " + " ".join([f"{k}:{to_item(v):.04f}" for k, v in sorted(dct.items())])
        return msg
    
    lprint(yaml.dump(cfg_dict, default_flow_style=False))

    module_path, module_name = cfg.train.module.split(':')
    cls =  getattr(importlib.import_module(module_path), module_name)
    train_dataloader, eval_dataloader, (model, optimizer, scheduler), loss_function, visualize = \
        Instantiate(cls, cfg.module)

    if eval_dataloader is None:
        train_dataloader, model, optimizer, scheduler = (
            accelerator.prepare(
                train_dataloader, 
                model, optimizer, scheduler
            )
        )
    else:
        train_dataloader, eval_dataloader, model, optimizer, scheduler = (
            accelerator.prepare(
                train_dataloader, eval_dataloader,
                model, optimizer, scheduler
            )
        )

    if resume:
        accelerator.load_state(resume)
        if skip_first > 0:
            train_dataloader = accelerator.skip_first_batches(train_dataloader, skip_first)

    init_trackers()
    start_time = time()
    for bi, batch in enumerate(tqdm(train_dataloader, desc='Training', disable=not accelerator.is_main_process)):
        if pass_loss_to_model:
            outputs = model(batch, loss_function=loss_function, accelerator=accelerator)
        else:
            outputs = model(batch)
        if 'loss_dict' in outputs:
            loss_dict = outputs['loss_dict']
        else:
            loss_dict = loss_function(batch, outputs)
            accelerator.backward(loss_dict['total'])

        if cfg.train.grad_clip > 0 and bi > cfg.train.grad_clip_start:
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)

        loss_dict['lr'] = optimizer.param_groups[0]['lr']
        loss_dict['gnorm'] = common.compute_grad_norm(model)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        log_metrics(**loss_dict)

        if bi % cfg.train.log_every == 0: 
            lprint(dict_to_msg(loss_dict), printer=tqdm.write)
        
        if bi % cfg.train.save_every == 0 and bi > 0:
            save_dir = work_dir / 'checkpoint'
            if save_model_by_step:
                save_dir = save_dir / f'step{bi}'
            accelerator.save_state(save_dir)
        
        if bi % cfg.train.eval_every == 0 and bi > 0:
            lprint('** Evaluation **', printer=tqdm.write)
            if eval_dataloader is not None:
                if accelerator.is_main_process and visualize is not None:
                    model.eval()
                    for batch in eval_dataloader:
                        with torch.no_grad():
                            outputs = model(batch)
                        figs = visualize(batch, outputs, work_dir, bi)
                        if vis_limit > 0 and isinstance(figs, list):
                            figs = figs[:vis_limit]
                        log_figs(figs)                   
                        break
                    model.train()
    
    save_dir = work_dir / 'checkpoint'
    if save_model_by_step:
        save_dir = save_dir / f'step{bi}'
    accelerator.save_state(save_dir)
    end_trackers()


if __name__ == "__main__":
    fire.Fire(main)