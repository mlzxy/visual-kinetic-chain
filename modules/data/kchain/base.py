import torch
import os.path as osp
from PIL import Image
from scipy.spatial.transform import Rotation
from pathlib import Path
import random
from torch import Tensor
import numpy as np
from typing import List
from torch.utils.data import Dataset
from torchvision import datasets
from dataclasses import dataclass, fields
import common


ENVS = {
'PAD_VALUE': 0
}


@dataclass
class VLSample:
    env: str = ""
    task: str = ""
    episode: float = 0
    episode_path: str = ""
    frame_id: float = 0
    lang: str = "" 

    lang_emb: Tensor = None # (L, C)

    rgb: Tensor = None # (3, H, W)

    skel_pts: Tensor = None # (H, M, 2)

    joints: Tensor = None # (H, J)
    gripper: Tensor = None # (H, x)
    xyz: Tensor = None # (H, 3)
    rpy: Tensor = None # (H, 3)
    end_points: Tensor = None # (H, 2)
    action: Tensor = None # (H, A)
    
    skel_hms: Tensor = None # (horizon, H, W)

    sample_id: int = -1
    skeleton: List = None # (points, edges)
    rgb_others: List = None #
    

class BaseVLDataset(Dataset):
    def __init__(self, transform, limits, horizon, min_points_per_edge=10, ordered=False, need_hms=False, multi_view=False, **kwargs):
        self._transform = transform
        self.limits = limits
        self.horizon = horizon
        self.min_points_per_edge = min_points_per_edge
        self.debug = False
        self.ordered = ordered
        self.need_hms = need_hms
        self.multi_view = multi_view
        self.include_other_frames = kwargs.get('include_other_frames', False)
    
    def get_sample_id(self, env_name, episode_id, frame_id):
        env_id = {
            'languagetable': 2,
            'calvin': 0,
            'rlbench': 1,
            'openx': 4,
            'rh20t': 3,
            'kuka': 5,
        }[env_name]
        return env_id * 50000 + episode_id * 5000 + frame_id

    
    def __getitem__(self, index) -> VLSample:
        raise NotImplementedError
    
    def __len__(self):
        return self.limits
    
    def combine_skeletons_in_horizon(self, skel_pts, image_hw, total_pad_size):
        image_height, image_width = image_hw
        if self.ordered:
            skel_pts, skel_pts_t = sample_horizon_points_from_skeletons(skel_pts, min_points_per_edge=self.min_points_per_edge, return_t=True)
        else:
            skel_pts = sample_horizon_points_from_skeletons(skel_pts, min_points_per_edge=self.min_points_per_edge)
        skel_pts -= (total_pad_size//2)
        if self.ordered:
            skel_pts = np.concatenate([skel_pts, skel_pts_t[..., None]], axis=-1)
        skel_pts = [self.filter_oob_points(pts, [image_height, image_width]) for pts in skel_pts]
        skel_pts = pad_stack(skel_pts) 
        return skel_pts

    def debug_visualize(self, rgb, skel_pts):
        if self.debug:
            try:
                imgs = []
                for pts in skel_pts:
                    img = common.draw_keypoints(rgb, pts, output_pil=True)
                    imgs.append(img)
                common.to_gif(imgs, file_path=Path(f'~/Desktop/{getattr(self, "debug_visname", "debug")}.gif').expanduser(), do_display=False)
            except:
                pass
    
    def load_lang_emb(self, episode_path, i=0):
        desc_emb_path = episode_path / 'description_embedding.pkl'
        if desc_emb_path.exists():
            desc_emb = common.load_pkl(desc_emb_path)
            feat = desc_emb['features'][i, :desc_emb['lengths'][i]]
            return feat 
        else:
            desc_emb = None
            return desc_emb
    
    def filter_oob_points(self, pts_xy, image_hw):
        H, W = image_hw
        mask = (pts_xy[:, :2].min(axis=1) >= 0) & (pts_xy[:, 0] <= W - 1) & (pts_xy[:, 1] <= H - 1)
        return pts_xy[mask]
   
    def transform(self, rgb, skel_pts):
        if self._transform is not None:
            H, M, _ = skel_pts.shape
            skel_frame_ids = np.arange(H).reshape(-1, 1).repeat(M, axis=1).flatten()

            if self.ordered:
                POINT_SIZE = 3
                skel_pts = skel_pts.reshape(-1, POINT_SIZE)
                keypoint_data = np.concatenate([skel_frame_ids[:, None], skel_pts[:, 2:]], axis=-1)
                _ = self._transform(image=rgb, keypoints=skel_pts[:, :2], keypoint_data=keypoint_data)
                rgb, n_skel_pts, keypoint_data = _['image'], np.array(_['keypoints']), np.array(_['keypoint_data'])
                n_skel_frame_ids = keypoint_data[:, 0]
                n_skel_pts = np.concatenate([n_skel_pts, keypoint_data[:, 1:]], axis=-1)
            else:
                POINT_SIZE = 2
                skel_pts = skel_pts.reshape(-1, POINT_SIZE)
                _ = self._transform(image=rgb, keypoints=skel_pts, keypoint_data=skel_frame_ids)
                rgb, n_skel_pts, n_skel_frame_ids = _['image'], np.array(_['keypoints']), np.array(_['keypoint_data'])

            new_skel_pts = [n_skel_pts[n_skel_frame_ids == i].reshape(-1, POINT_SIZE) for i in range(H)]
            new_skel_pts = [self.filter_oob_points(pts, rgb.shape[:2]) for pts in new_skel_pts] # keep the valid ones
            new_skel_pts = pad_stack(new_skel_pts) # H, M, POINT_SIZE

            if self.ordered:
                new_skel_pts[:, :, -1] /= new_skel_pts[:, :, -1].max()
            return rgb, new_skel_pts
            
        return rgb, skel_pts
    
    def render_skel_hms(self, rgb, skel_pts):
        if self.need_hms:
            size = 5
            sigma = 3
            hm = np.zeros((self.horizon, ) + rgb.shape[:2], dtype=np.float32)
            if not hasattr(self, 'cache_coords'):
                ys = np.arange(rgb.shape[0])
                xs = np.arange(rgb.shape[1]) 
                xs, ys = np.meshgrid(xs, ys) 
                coords = np.concatenate([ys[..., None], xs[..., None]], axis=-1)
            else:
                coords = self.cache_coords
            
            def gaussian_kernel(arr, center):
                v = np.exp(-np.sum((arr - center) ** 2, axis=-1)/(2*sigma**2))
                v /= v.max()
                return v
        
            for i in range(skel_pts.shape[0]):
                for j in range(skel_pts.shape[1]):
                    pt = skel_pts[i, j].round().astype(int)
                    wc, hc = pt[0], pt[1]
                    hm[i, max(hc - size, 0):hc+size, max(wc - size, 0):wc+size] = \
                        np.maximum(hm[i, max(hc - size, 0):hc+size, max(wc - size, 0):wc+size], 
                                gaussian_kernel(coords[max(hc - size, 0):hc+size, max(wc - size, 0):wc+size], 
                                                np.array([hc, wc]).reshape(1, 1, -1)))
            
            if self.debug:
                rgb = rgb.copy()
                rgb[hm[0] > 0.5, :] = np.array([255, 0, 0])
                Image.fromarray(rgb).save(osp.expanduser('~/Desktop/debug.png'))
            
            return hm

    
    
def pad_stack(tensors, strategy='random', return_mask=False, mask_last=True):
    max_H = max([t.shape[0] for t in tensors])
    rest_shape = list(tensors[0].shape[1:])
    B = common.backend(tensors[0])
    padded_tensors, padded_masks = [], []
    for tensor in tensors:
        pad_size = max_H - tensor.shape[0]
        if pad_size > 0:
            if tensor.shape[0] > 0:
                if strategy == 'random':
                    brng = torch if isinstance(tensor, torch.Tensor) else np.random
                    random_indices = brng.randint(0, tensor.shape[0], (pad_size,))
                elif strategy == 'last':
                    random_indices = [tensor.shape[0] - 1,] * pad_size
                else:
                    raise KeyError(f'pad_stack strategy: {strategy}')
                random_values = tensor[random_indices]
                padded_tensor = common.cat([tensor, random_values], 0)
            else:
                padded_tensor = B.zeros([max_H, ] + rest_shape)
        else:
            padded_tensor = tensor

        padded_tensors.append(padded_tensor)
        if return_mask:
            padded_mask = B.ones([tensor.shape[0],])
            if mask_last: padded_mask[-1] = 0
            padded_mask = common.cat([padded_mask, B.zeros([pad_size, ])], 0)
            padded_masks.append(padded_mask)
    
    stacked_tensors = B.stack(padded_tensors)
    if return_mask:
        return stacked_tensors, B.stack(padded_masks)
    else:
        return stacked_tensors


def center_pad_image(image, return_pad=False, pad_value=None):
    if pad_value is None: pad_value = ENVS['PAD_VALUE']
    height, width = image.shape[:2]
    max_dim = max(height, width)
    pad_height = max_dim - height
    pad_width = max_dim - width
    top_pad = pad_height // 2
    bottom_pad = pad_height - top_pad
    left_pad = pad_width // 2
    right_pad = pad_width - left_pad
    padded_image = np.pad(image, ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)), mode='constant', constant_values=(pad_value,))
    if return_pad:
        return padded_image, (top_pad, left_pad)
    else:
        return padded_image


def normalize_rad(a):
    """normalize radian to [-1, 1]"""
    a = np.copy(a)
    a[a > np.pi] -= (2 * np.pi)
    a[a < -np.pi] += (2 * np.pi)
    a /= np.pi
    return a

def denormalize_rad(a):
    return a * np.pi


def sensitive_gimble_fix(euler):
    """
    :param euler: euler angles in degree as np.ndarray in shape either [3] or
    [b, 3]
    """
    # selecting sensitive angle, y-axis
    select1 = (89 < euler[..., 1]) & (euler[..., 1] < 91)
    euler[select1, 1] = 90
    # selecting sensitive angle
    select2 = (-91 < euler[..., 1]) & (euler[..., 1] < -89)
    euler[select2, 1] = -90

    # recalulating the euler angles, see assert
    r = Rotation.from_euler("xyz", euler, degrees=True)
    euler = r.as_euler("xyz", degrees=True)

    select = select1 | select2
    assert (euler[select][..., 2] == 0).all(), euler # z-axis for the fixed ones
    return euler


def obtain_skeletons(key_points, key_point_names, named_connections): 
    """
    key_points: (N, 2)
    key_point_names: str x N
    named_connections: (str, str) x E

    return points (N', 2), edges (E, 2)
    """
    all_names = set()
    for a, b in named_connections:
        all_names.add(a)
        all_names.add(b)
    all_names = list(all_names)

    points = []
    for n in all_names:
        points.append(key_points[key_point_names.index(n)])
    points = np.stack(points)

    edges = []
    for a, b in named_connections:
        edges.append((
            all_names.index(a),
            all_names.index(b)
        )) 
    return points, np.array(edges)


def sample_horizon_points_from_skeletons(horizon_skeletons, min_points_per_edge=10, return_t=False):
    """
    this function assumes all points and edges are valid, be sure to filter them 
    before calling

    horizon_skeletons: [(n_points, 2), (n_edges, 2)] x H
    returns: (H, M, 2)
    """ 
    horizon_skeletons = [skel if isinstance(skel[0], np.ndarray) 
                         else (np.array(skel[0]), np.array(skel[1])) 
                        for skel in horizon_skeletons]
    
    max_M = max([len(skel[1]) * min_points_per_edge for skel in horizon_skeletons])
    output = np.zeros([len(horizon_skeletons), max_M, 2])
    output_t = np.zeros([len(horizon_skeletons), max_M])
    
    for i, (points, edges) in enumerate(horizon_skeletons):
        points = points.reshape(-1, 2)
        M = len(edges) * min_points_per_edge
        leftover = max_M - M
        start = points[edges[:, 0]]  # (n_edges, 2)
        end = points[edges[:, 1]]   # (n_edges, 2)
        t = np.linspace(0, 1, min_points_per_edge).reshape(-1, 1, 1)
        sampled_points = ((end - start)[None, ...] * t + start[None, ...]).reshape(-1, 2)
        output[i, :M] = sampled_points
        t = (t.reshape(-1, 1, 1) + np.arange(0, len(edges)).astype(np.float32).reshape(1, -1, 1)).flatten()
        if leftover > 0:
            lengths = np.linalg.norm(end - start, axis=1)
            lengths /= lengths.sum()
            t_leftover = []
            for j in range(leftover):
                edge_id = np.random.choice(len(edges), size=1, p=lengths)
                e_start, e_end = start[edge_id], end[edge_id]
                rn = random.random()
                point = (e_end - e_start) * rn + e_start
                output[i, M + j, :] = point
                t_leftover.append(edge_id + rn)
            t = np.concatenate([t, np.array(t_leftover).flatten()])
        output_t[i] = t
    if return_t:
        return output, output_t
    else:
        return output

    
class VLCollator:
    def __init__(self, stack_actions=False, stack_skels=True):
        self.stack_actions = stack_actions
        self.stack_skels = stack_skels
    
    def __call__(self, samples: List[VLSample]):
        envs, langs, episodes, frame_ids, lang_embs, rgbs, skel_pts, joints, grippers, xyzs, rpys, end_points, actions = [list() for i in range(13)] 

        skel_hms = []
        sample_ids = []

        def dispatch(sample):
            if sample.sample_id >= 0:
                sample_ids.append(sample.sample_id)

            # must have attributes
            envs.append(sample.env)
            episodes.append(sample.episode)
            frame_ids.append(sample.frame_id)
            langs.append(sample.lang)
            lang_embs.append(torch.from_numpy(sample.lang_emb))
            rgbs.append(torch.from_numpy(sample.rgb).permute(2, 0, 1))
            skel_pts.append(torch.from_numpy(sample.skel_pts).permute(1, 0, 2))  # H, M, 2 -> M, H, 2
            actions.append(torch.from_numpy(sample.action))

            if sample.gripper is not None:
                grippers.append(torch.from_numpy(sample.gripper))

            if sample.xyz is not None:
                xyzs.append(torch.from_numpy(sample.xyz))
            
            if sample.rpy is not None:
                rpys.append(torch.from_numpy(sample.rpy))
            
            if sample.end_points is not None:
                end_points.append(torch.from_numpy(sample.end_points))
            
            if sample.skel_hms is not None:
                skel_hms.append(torch.from_numpy(sample.skel_hms))



        for sample in samples:
            if isinstance(sample, list):
                for s in sample: dispatch(s)
            else:
                dispatch(sample)
        
        lang_embs, lang_emb_masks = pad_stack(lang_embs, strategy='last', return_mask=True)
        # lang_embs, lang_emb_masks = lang_embs[:, :-1], lang_emb_masks[:, :-1]

        return dict(envs=envs, episodes=episodes, frame_ids=frame_ids,
                    langs=langs, rgbs=torch.stack(rgbs), lang_embs=lang_embs,
                    lang_emb_masks=lang_emb_masks, actions=torch.stack(actions)  if self.stack_actions else actions,

                    sample_ids=None if len(sample_ids) == 0 else torch.as_tensor(sample_ids),

                    xyzs=torch.stack(xyzs) if self.stack_actions and len(xyzs) > 0 else xyzs,                    
                    grippers=torch.stack(grippers) if self.stack_actions and len(grippers) > 0 else grippers,                    
                    rpys=torch.stack(rpys) if self.stack_actions and len(rpys) > 0 else rpys,                    
                    

                    skel_hms=torch.stack(skel_hms) if len(skel_hms) > 0 else None,

                    skel_pts=pad_stack(skel_pts).permute(0, 2, 1, 3) if self.stack_skels else None, # N, M, H, 2 -> N, H, M, 2
                    end_points=torch.stack(end_points) if len(end_points) > 0 and self.stack_skels else None)




class MultiDataset(Dataset):
    def __init__(self, datasets: List[BaseVLDataset], probability=None, limits=1000):
        super().__init__()
        self.datasets = datasets
        self.limits = limits
        if probability is None:
            self.probability = np.ones([len(datasets)], dtype=np.float32) / len(datasets) 
        else:
            assert len(probability) == len(datasets)
            self.probability = np.array(probability) / np.sum(probability) 
    
    def __len__(self):
        return self.limits

    def __getitem__(self, index) -> VLSample:
        ind = int(np.random.choice(len(self.datasets), size=1, p=self.probability))
        return self.datasets[ind][index]


def query_next_kf(i, kps):
    for k in kps:
        if k > i:
            return k
    return i