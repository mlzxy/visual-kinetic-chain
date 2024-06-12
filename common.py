import pickle
import itertools
from PIL import Image
from pathlib import Path
import sys
from scipy.spatial.transform import Rotation, Slerp
import os.path as osp
from torch import nn
import ujson as json
import numpy as np
import torch 
from torchvision.utils import Optional, Tuple, Union, ImageDraw, List, Image
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
import ast
from easydict import EasyDict as dict_to_class
from torch import nn, Tensor, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from modules.optim.warmup import GradualWarmupScheduler


def assign_to_class(cfg, opts):
    for ks, v in opts.items():
        ks = ks.split('.')
        o = cfg
        for k in ks[:-1]:
            o = getattr(o, k)
        setattr(o, ks[-1], v)
    return cfg
             

def add_default(new_cfg, default_cfg):
    for k, v in vars(default_cfg).items():
        if not k.startswith('__'):
            if hasattr(new_cfg, k):
                if isinstance(getattr(new_cfg, k), type):
                    add_default(getattr(new_cfg, k), v)
            else:
                setattr(new_cfg, k, v)
    return new_cfg  

def class_to_dict(T):
    if isinstance(T, type):
        return {k: class_to_dict(v) for k, v in vars(T).items() if not k.startswith('__')}
    else:
        return T

def load_pkl(fp):
    with open(fp, 'rb') as f:
        return pickle.load(f)

def dump_pkl(fp, obj):
    with open(fp, 'wb') as f:
        pickle.dump(obj, f)

def dump_json(fp, obj):
    with open(fp, 'w') as f:
        json.dump(obj, f)

def load_json(fp):
    with open(fp) as f:
        return json.load(f)

def backend(a):
    if isinstance(a, torch.Tensor):
        return torch 
    else:
        return np


def cat(ts, axis=0):
    if isinstance(ts[0], torch.Tensor):
        return torch.cat(ts, axis)
    else:
        return np.concatenate(ts, axis)


def mean_list(lst):
    return sum(lst) / len(lst)

    
def timestamp():
    import time 
    return int(time.time())


def to_gif(img_list, file_path=None, frame_duration=0.1, do_display=True):
    from moviepy.editor import ImageClip, concatenate_videoclips, ipython_display
    import IPython.display as ipd

    clips = [ImageClip(np.array(img)).set_duration(frame_duration) for img in img_list]

    clip = concatenate_videoclips(clips, method="compose", bg_color=(255, 255, 255))

    if file_path is not None:
        method = {
            'gif': clip.write_gif,
            'mp4': clip.write_videofile,
            'img': clip.write_images_sequence
        }
        Path(file_path).parent.mkdir(exist_ok=True, parents=True)
        ext = osp.splitext(file_path)[1][1:]
        method[ext](str(file_path), fps=24, verbose=False, logger=None)

    if do_display:
        src = clip if file_path is None else file_path
        ipd.display(ipython_display(str(src), fps=24, rd_kwargs=dict(logger=None), autoplay=1, loop=1))



@torch.no_grad()
def draw_keypoints(
    image: torch.Tensor,
    keypoints: torch.Tensor,
    connectivity: Optional[List[Tuple[int, int]]] = None,
    colors: Optional[Union[str, Tuple[int, int, int]]] = (255, 0, 0),
    line_color = 'white',
    radius: int = 2,
    width: int = 3,
    output_pil=True,
    transparency=1.0,
    line_under=True
) -> torch.Tensor:
    def is_valid(*args):
        return all([a >= 0 for a in args])
    
    if isinstance(image, Image.Image):
        image = pil_to_tensor(image)
    
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)
        if image.shape[-1] == 3:
            image = image.permute(2, 0, 1)
    
    POINT_SIZE = keypoints.shape[-1]
    if isinstance(keypoints, np.ndarray):
        keypoints = torch.from_numpy(keypoints)
    
    keypoints = keypoints.reshape(1, -1, POINT_SIZE)

    ndarr = image.permute(1, 2, 0).cpu().numpy()
    img_to_draw = Image.fromarray(ndarr)
    if transparency < 1.0:
        draw = ImageDraw.Draw(img_to_draw, 'RGBA')
    else:
        draw = ImageDraw.Draw(img_to_draw, None if POINT_SIZE == 2 else 'RGBA')
    keypoints = keypoints.clone()
    if POINT_SIZE == 3:
        keypoints[:, :, -1] *= 255
    img_kpts = keypoints.to(torch.int64).tolist()

    for kpt_id, kpt_inst in enumerate(img_kpts):
        kpt_size = len(kpt_inst[0])

        def draw_line():
            if connectivity is not None:
                for connection in connectivity:
                    start_pt_x = kpt_inst[connection[0]][0]
                    start_pt_y = kpt_inst[connection[0]][1]

                    end_pt_x = kpt_inst[connection[1]][0]
                    end_pt_y = kpt_inst[connection[1]][1]

                    if not is_valid(start_pt_x, start_pt_y, end_pt_x, end_pt_y):
                        continue

                    if transparency < 1.0:
                        kp_line_color = line_color + (int(255*(1- transparency)), )
                    else:
                        kp_line_color = line_color

                    draw.line(
                        ((start_pt_x, start_pt_y), (end_pt_x, end_pt_y)),
                        width=width, fill=kp_line_color
                    )
        
        def draw_points():
            for inst_id, kpt in enumerate(kpt_inst):
                if not is_valid(*kpt):
                    continue
                x1 = kpt[0] - radius
                x2 = kpt[0] + radius
                y1 = kpt[1] - radius
                y2 = kpt[1] + radius
                if len(kpt) == 3:
                    kp_color = colors + (int(kpt[2]), )
                elif transparency < 1.0:
                    kp_color = colors + (int(255*(1- transparency)), )
                else:
                    kp_color = colors
                draw.ellipse([x1, y1, x2, y2], fill=kp_color, outline=None, width=0)
        
        if line_under:
            draw_line()
            draw_points()
        else:
            draw_points()
            draw_line()
            
    if output_pil:
        return img_to_draw  
    else:
        return torch.from_numpy(np.array(img_to_draw)).permute(2, 0, 1).to(dtype=torch.uint8)


    
def compute_grad_norm(model):
    if isinstance(model, nn.Module):
        grads = [
            param.grad.detach().flatten()
            for param in model.parameters()
            if param.grad is not None
        ]
    else:
        grads = [param.grad.detach().flatten() for param in model if param.grad is not None]
    norm = torch.cat(grads).norm()
    return norm



def get_bc_loss(preds, labels, reduction='mean'):
    # sim = nn.CosineSimilarity(dim=-1)(preds, labels)
    return {
        'mse': nn.MSELoss(reduction=reduction)(preds, labels),
        'L1': nn.L1Loss(reduction=reduction)(preds, labels),
        # 'cosine': 1.0 - sim
    }


def average_loss_dict_by_horizon(loss_dict, dim=1, final_reduction='mean'):
    total_dict = {}
    log_dict = {}
    for k, v in loss_dict.items():
        losses = v.mean(dim=list(set(range(len(v.shape)))  - {dim}))
        for i, lv in enumerate(losses):
            log_dict[k + ':' + str(i).zfill(2)] = lv
        total_dict[k] = losses.mean() if final_reduction == 'mean' else losses.sum()
    return {**log_dict, **total_dict}, total_dict



def add_prefix_to_dict(prefix, dct):
    return {prefix + k: v for k, v in dct.items()}

    

def load_model(config_file, checkpoint=None, device='cpu', transform_kwargs=None, return_config=False):
    if not config_file.endswith('cfg.json') and checkpoint is None:
        checkpoint =  osp.join(str(config_file), 'checkpoint/model.safetensors')
        config_file = osp.join(str(config_file), 'cfg.json')

    from require import require
    import importlib
    from modules.template import Instantiate
    from modules.data.augmentation import image_aug_recipe
    cfg = require(config_file, toplevel=True)
    module_path, module_name = cfg.train.module.split(':')
    cls = getattr(importlib.import_module(module_path), module_name)
    action_model, _, _ = Instantiate(cls, cfg.module, ['model'])[0]
    state_dict = load_safetensors(checkpoint)
    action_model.load_state_dict(state_dict)
    action_model.to(device)
    action_model.eval()
    transform_kwargs = transform_kwargs or {'target_size': 224}
    transform = image_aug_recipe(**transform_kwargs, color=0, spatial=0, hflip=0, vflip=0, 
                     need_skeleton=False, normalize=True, to_tensor=True)
    
    if return_config:
        return action_model, transform, cfg
    else:
        return action_model, transform


def load_clip(device='cpu'):
    from transformers import CLIPProcessor, CLIPModel
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    clip = clip.to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    return clip, processor



def load_safetensors(path):
    from safetensors import safe_open
    tensors = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def load_hydra_cfg(config_path, overrides=[]):

    from hydra import compose, initialize
    from omegaconf import OmegaConf

    with initialize(config_path=osp.relpath(osp.dirname(config_path), start=osp.dirname(__file__)), job_name="load_config"):
        cfg = compose(config_name=osp.splitext(osp.basename(config_path))[0], overrides=overrides)
    return cfg



def get_adam_with_cosine_lr_schedule(model, lr, weight_decay, steps, warmup_steps):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    after_scheduler = CosineAnnealingLR(optimizer, T_max=steps, eta_min=lr / 100)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_steps, after_scheduler=after_scheduler)
    return optimizer, scheduler


def freeze_module(mod: nn.Module):
    for p in mod.parameters():
        p.requires_grad = False
    
    
def get_learnable_params(mod: nn.Module):
    lst = []
    for k, v in mod.named_parameters():
        if v.requires_grad:
            lst.append(k) 
    return lst


def wrap_stdout(log_file_path):
    Path(log_file_path).parent.mkdir(parents=True, exist_ok=True)

    class Logger(object):
        def __init__(self, *writers):
            self.writers = writers

        def write(self, message):
            for writer in self.writers:
                writer.write(message)

        def flush(self):
            for writer in self.writers:
                writer.flush()

    try:
        log_file = open(log_file_path, 'w')  # Open log file in append mode
    except IOError as e:
        print("Error opening log file:", e)
        return

    original_stdout = sys.stdout
    sys.stdout = Logger(sys.stdout, log_file)

    # Function to restore the original stdout
    def restore_stdout():
        sys.stdout = original_stdout
        log_file.close()

    return restore_stdout


def interpolate_quat(l_w, l_quat, r_w, r_quat, output_format='quat', input_format='quat'):
    if input_format == 'quat':
        L = Rotation.from_quat(l_quat)
        R = Rotation.from_quat(r_quat)
    else:
        L = Rotation.from_euler("xyz", l_quat)
        R = Rotation.from_euler("xyz", r_quat)
    slerp = Slerp([0, 1], Rotation.concatenate([L, R]))
    result = slerp(1 - l_w)

    if output_format == 'quat':
        return result.as_quat()
    else:
        return result.as_euler("xyz")


def np_stack(lst_arr):
    if len(lst_arr) == 0:
        return None
    else:
        return np.stack(lst_arr)
    

    

def subsample_or_pad(lst, L):
    if len(lst) == L:
        return lst
    elif len(lst) > L:
        lst = np.linspace(lst[0], lst[-1], L)
        lst = lst.astype(int).tolist()
        return lst
    else:
        padded_lst = lst + [lst[-1]] * (L - len(lst))
        return padded_lst


def discretize_euler(euler, resolution=5):
    euler = euler + 180
    assert np.min(euler) >= 0 and np.max(euler) <= 360
    disc = np.around((euler / resolution)).astype(int)
    disc[disc == int(360 / resolution)] = 0
    return disc


def quaternion_to_discrete_euler(quaternion, resolution=5):
    euler = Rotation.from_quat(quaternion).as_euler('xyz', degrees=True)
    disc = discretize_euler(euler, resolution)
    return disc

def discrete_euler_to_quaternion(discrete_euler, resolution=5):
    euluer = (discrete_euler * resolution) - 180
    return Rotation.from_euler('xyz', euluer, degrees=True).as_quat()




# hm = generate_heatmap_from_screen_pts(torch.from_numpy(keypoints), (220, 300))
# # points in xy, but res in hw
def generate_heatmap_from_screen_pts(pt, res, sigma=1.5, thres_sigma_times=3):
    """
    Pytorch code to generate heatmaps from point. Points with values less than
    thres are made 0
    :type pt: torch.FloatTensor of size (num_pt, 2)
    :type res: int or (int, int), meaning (h,w)
    :param sigma: the std of the gaussian distribition. if it is -1, we
        generate a hm with one hot vector
    :type sigma: float
    :type thres: float
    """
    num_pt, x = pt.shape
    assert x == 2
    assert sigma > 0

    if isinstance(res, int):
        resx = resy = res
    else:
        resx, resy = res

    _hmx = torch.arange(0, resy).to(pt.device)
    _hmx = _hmx.view([1, resy]).repeat(resx, 1).view([resx, resy, 1])
    _hmy = torch.arange(0, resx).to(pt.device)
    _hmy = _hmy.view([resx, 1]).repeat(1, resy).view([resx, resy, 1])
    hm = torch.cat([_hmx, _hmy], dim=-1)
    hm = hm.view([1, resx, resy, 2]).repeat(num_pt, 1, 1, 1) # one HxW heatmap for each point?

    pt = pt.view([num_pt, 1, 1, 2])
    hm = torch.exp(-1 * torch.sum((hm - pt) ** 2, -1) / (2 * (sigma**2))) # RBF Kernel
    thres = np.exp(-1 * (thres_sigma_times**2) / 2) # truncated
    hm[hm < thres] = 0.0

    hm /= torch.sum(hm, (1, 2), keepdim=True) + 1e-6 # normalization
    return hm 


def sort_2d_points(points_bh2, N=1000):
    points_v = points_bh2[:, :, 1] + points_bh2[:, :, 0] * N
    order = torch.argsort(points_v, dim=1)
    bs, n_p = points_bh2.shape[:2]
    result = points_bh2[torch.arange(0, bs, device=points_bh2.device).reshape(-1, 1).repeat(1, n_p).flatten(), order.flatten(), :].reshape(bs, n_p, -1)
    return result


def iround(x): return int(round(x))


def resize_to_closest(img, dividable=14):
    from torchvision.transforms import functional as tvF

    if img.shape[0] == 3:
        h, w = img.shape[1:]
    else:
        h, w = img.shape[:2]
    h, w = max(iround(h / dividable), 1) * dividable, max(iround(w / dividable), 1) * dividable

    if img.shape[0] == 3:
        return tvF.resize(img, (h, w), interpolation=tvF.InterpolationMode.BICUBIC)
    else:
        img = Image.fromarray(img)
        img = img.resize((w, h), Image.Resampling.BICUBIC)
        return np.array(img)
    
    


def get_match_loss(pred, label, loss_func, preprocess=lambda a: a, label_mask=None):
    """ assuming the pred is B x H x C"""
    b_h = pred.shape[:2]
    indices = list(range(pred.shape[1]))
    permutations = list(itertools.permutations(indices))

    losses = []

    for inds in permutations:
        loss = loss_func(preprocess(pred[:, inds, :]), label)
        if len(loss.shape) == 3:
            loss = loss.sum(dim=-1)
        elif len(loss.shape) == 1:
            loss = loss.reshape(b_h)
        
        if label_mask is not None:
            loss *= label_mask
        
        loss = loss.reshape(b_h + (1, ))
        losses.append(loss)

    losses = torch.cat(losses, dim=-1)
    return losses.min(dim=-1).values

    


def reduce_params_to_0(*modules):
    zero = 0
    for mod in modules:
        for p in mod.parameters():
            zero += (p.sum() * 0)
    return zero


def to_red_heatmap(heatmap, normalize=False):
    if len(heatmap.shape) == 2:
        heatmap = heatmap[None, ...]
    if normalize:
        heatmap = heatmap / heatmap.max()
    heatmap = torch.cat((heatmap, torch.zeros(2, *heatmap.shape[1:], device=heatmap.device)))
    return heatmap
