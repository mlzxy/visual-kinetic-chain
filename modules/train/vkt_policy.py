import torch
import numpy as np
from collections import defaultdict, Counter
import common
from pathlib import Path
from torch import nn, Tensor, optim
import modules.template as T
from modules.data import KChainDataLoader
from geomloss import SamplesLoss
from modules.data.augmentation import denormalize_bchw_image
from transformers import CLIPProcessor, CLIPModel
from transformers.modeling_outputs import BaseModelOutputWithPooling
from peft import LoraConfig, get_peft_model
from torch.optim.lr_scheduler import CosineAnnealingLR
from modules.optim.warmup import GradualWarmupScheduler
from modules.model.pos_emb import PositionEmbeddingRandom
from modules.model.two_way_transformer_mv import TwoWayTransformer, Attention
from modules.data.kchain.sim import RLBenchDataset, CalvinDataset
from typing import List, Dict


class Config:
    class data:
        pass

    class model:
        n_points: int = 200
        horizon = 10
        hidden_size = 512
        depth = 6

        lora_modules: List[str] = ["q_proj", "v_proj"]
        lora_kwargs: Dict = {}

    class learn:
        lr: float = 0.001 
        weight_decay: float = 0.001
        steps: int = 10000
        warmup_steps: int = 1000


class LanguageSkelPolicy(nn.Module):
    def __init__(self, n_points=200, horizon=10, hidden_size=512,
                 depth=6, output_feats=False,
                 lora_modules=["q_proj", "v_proj"], lora_kwargs={}):
        super().__init__()
        point_hidden_size = 8192
        clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        lora_config = LoraConfig(
            target_modules=lora_modules,
            **lora_kwargs
        )
        self.backbone = get_peft_model(clip.vision_model, lora_config)

        self.n_points = n_points
        self.horizon = horizon
        self.hidden_size = hidden_size

        self.query_embeddings = nn.Embedding(horizon, hidden_size)

        self.position_embedding = PositionEmbeddingRandom(hidden_size // 2) 
        self.visual_projection = nn.Linear(768, hidden_size, bias=False)
        self.text_projection = nn.Linear(512, hidden_size, bias=False)

        self.bi_transformer = TwoWayTransformer(depth=depth, embedding_dim=hidden_size,
                            num_heads=8, mlp_dim=2048, multi_view=True, multi_view_visual=True)
        
        self.next_point_mlp = nn.Sequential(
            nn.Linear(hidden_size, point_hidden_size),
            nn.GELU(),
            nn.Linear(point_hidden_size, n_points*2) 
        )

    def multi_view_forward(self, inputs, num_views):
        bs, dev = len(inputs['rgbs']), inputs['rgbs'].device
        iH, iW = inputs['rgbs'].shape[2:]
        fH, fW = iH // 16, iW // 16

        out: BaseModelOutputWithPooling = self.backbone(pixel_values=inputs['rgbs'])

        visuals = out.last_hidden_state[:, 1:, :]
        visuals = self.visual_projection(visuals)  # B, L, C

        visuals = visuals.permute(0, 2, 1).reshape(bs, self.hidden_size, fH, fW)
        pos_emb = self.position_embedding([fH, fW])[None, ...]

        texts = inputs['lang_embs']  # B, tL, C
        texts_mask = inputs['lang_emb_masks']  # B, tL
        texts = self.text_projection(texts)

        query = torch.cat([self.query_embeddings.weight.reshape(1, self.horizon, -1).repeat(bs, 1, 1),  texts], dim=1)
        texts_mask = torch.cat([torch.ones([bs, self.horizon], device=dev), texts_mask], dim=1)
        
        query_after, visuals_after = self.bi_transformer(image_embedding=visuals, image_pe=pos_emb, 
                                                        query_embedding=query, query_mask=texts_mask, 
                                                        num_views=num_views)
        query_after = query_after[:, :self.horizon] # B, H, C

        if self.next_point_mlp is not None:
            pts_logits = self.next_point_mlp(query_after) # B, H, N_Points * 2
            pts = torch.tanh(pts_logits).reshape(bs, self.horizon, self.n_points, 2)
        else:
            pts = None
        output = {'skel_pts': pts, 'query': query_after}
        return output

    
    def forward(self, inputs, loss_function=None, acceleartor=None):
        env_to_indices = defaultdict(list)
        for i, e in enumerate(inputs['envs']):
            env_to_indices[e].append(i)
        
        env_to_views = {
            'calvin': CalvinDataset.NUM_VIEWS,
            'rlbench': RLBenchDataset.NUM_VIEWS
        }
        full_indices = []
        output_skel_pts = []
        output_query = []

        for e, indices in env_to_indices.items():
            num_views = env_to_views[e]

            sample_ids = inputs['sample_ids'][indices].reshape(-1, num_views)
            assert torch.all(sample_ids == sample_ids[:, :1])

            output = self.multi_view_forward(
                {'rgbs': inputs['rgbs'][indices], 'lang_embs': inputs['lang_embs'][indices],
                'lang_emb_masks': inputs['lang_emb_masks'][indices]}, num_views)

            output_skel_pts.append(output['skel_pts'])
            output_query.append(output['query'])
            full_indices += indices

        if self.next_point_mlp is not None:
            skel_pts = torch.cat(output_skel_pts)
            skel_pts = skel_pts[np.argsort(full_indices)]
        else:
            skel_pts = None
        query = torch.cat(output_query)
        query = query[np.argsort(full_indices)]

        output = {'skel_pts': skel_pts, 'query': query} 
        return output
        

class LANG_SKEL_POLICY(KChainDataLoader, T.Mixin):
    def __init__(self, cfg: Config):
        KChainDataLoader.__init__(self, cfg.data)
        self.model_cfg = cfg.model
        self.learn_cfg = cfg.learn
        
    @property
    def model(self):
        mcfg = self.model_cfg
        lcfg = self.learn_cfg
        model = LanguageSkelPolicy(n_points=mcfg.n_points, horizon=mcfg.horizon, hidden_size=mcfg.hidden_size,
                    depth=mcfg.depth, lora_modules=mcfg.lora_modules, lora_kwargs=mcfg.lora_kwargs)
        
        optimizer = optim.AdamW(model.parameters(), lr=lcfg.lr, weight_decay=lcfg.weight_decay)
        after_scheduler = CosineAnnealingLR(optimizer, T_max=lcfg.steps, eta_min=lcfg.lr / 100)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=lcfg.warmup_steps, after_scheduler=after_scheduler)
        return model, optimizer, scheduler

    @property
    def loss_function(self):
        """ return a loss_dict with at least a key `total`"""
        sample_loss = SamplesLoss(loss="sinkhorn", p=2)
        n_points = self.model_cfg.n_points
        L1 = nn.L1Loss()
        MSE = nn.MSELoss()


        def function(batch, outputs)-> Dict[str, float]:
            bs = len(batch['rgbs'])
            H, W = batch['rgbs'].shape[2:]
            label_skel_pts = batch['skel_pts'].clone().float() # B, horizon, M, C
            label_skel_pts[..., 0] /= W
            label_skel_pts[..., 1] /= H
            label_skel_pts -= 0.5
            label_skel_pts *= 2
            horizon = label_skel_pts.shape[1]
            losses = []
            loss_dict = {}

            for i in range(horizon):
                loss = sample_loss(outputs['skel_pts'][:, i].contiguous(), label_skel_pts[:, i].contiguous())
                valid_mask = ~torch.all(label_skel_pts[:, i].reshape(bs, -1) == 0, dim=1)
                loss = loss[valid_mask]
                if len(loss) > 0:
                    _ = loss.mean()
                    losses.append(_)
                    loss_dict[f'l{str(i).zfill(2)}'] = _

            loss_dict['total'] = common.mean_list(losses) 
            return loss_dict
        
        return function


    @property
    def visualize(self):
        import wandb
        
        def function(batch, outputs, work_dir: Path, step):
            H, W = batch['rgbs'].shape[2:]
            rgbs = denormalize_bchw_image(batch['rgbs']).to(torch.uint8)
            skel_pts = outputs['skel_pts'].clone()
            skel_pts_target = batch['skel_pts']
            horizon = skel_pts.shape[1]
            skel_pts /= 2
            skel_pts += 0.5
            skel_pts[..., 0] *= W
            skel_pts[..., 1] *= H

            out_dir = Path(work_dir) / f'vis/step{step}'
            out_dir.mkdir(parents=True, exist_ok=True)

            result = []
            for i in range(len(rgbs)):
                env = batch['envs'][i]
                lang = batch['langs'][i]
                pred_figs = []
                label_figs = []
                for j in range(horizon):
                    fig = common.draw_keypoints(rgbs[i], skel_pts[i, j])
                    fig_label = common.draw_keypoints(rgbs[i], skel_pts_target[i, j])
                    pred_figs.append(fig)
                    label_figs.append(fig_label)

                pred_mp4 = out_dir / f'{i}.mp4'
                label_mp4 = out_dir / f'{i}.label.mp4'

                common.to_gif(pred_figs, file_path=pred_mp4, do_display=False)                
                common.to_gif(label_figs, file_path=label_mp4, do_display=False)

                result.append(wandb.Video(str(pred_mp4), caption=f'{env}:{lang} (pred {i})'))
                result.append(wandb.Video(str(label_mp4), caption=f'{env}:{lang} (label {i})'))
            return result 

        return function
    
    
