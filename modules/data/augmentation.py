import albumentations as A
import cv2
import torch
from albumentations.pytorch import ToTensorV2

def image_aug_recipe(target_size = None, color=0.25, spatial=0.25, hflip=0.5, vflip=0.1, mask=0, rotate=True, perspective=True,
                     need_skeleton=False, normalize=True, to_tensor=False, remove_invisible=True):
    T_flip = A.Compose([
        A.HorizontalFlip(p=hflip),
        A.VerticalFlip(p=vflip)
    ])

    T_mask = A.XYMasking(
        num_masks_x=10, 
        num_masks_y=10,  
        mask_x_length=6,  
        mask_y_length=6, 
        fill_value=0,  
        mask_fill_value=0,  
        always_apply=False, 
        p=mask
    )

    T_spatial = [A.Affine(rotate=(0, 0), translate_percent=(-0.3, 0.3), scale=(0.5, 0.95))]
    
    if perspective:
        T_spatial.append(A.Perspective(scale=(0.2, 0.4)))
    
    if rotate:
        T_spatial.append(A.Rotate(limit=60, value=0, border_mode=cv2.BORDER_CONSTANT))

    T_spatial = A.OneOf(T_spatial, p=spatial)

    T_color = A.OneOf([
            # A.ElasticTransform(alpha=80.0),
            # A.GaussianBlur(blur_limit=(3, 5), sigma_limit=(0.1, 5.)),
            # A.Posterize(num_bits=2, p=1.0),

            A.ColorJitter(
                brightness=(0.5, 1),  # Union[float, Tuple[float, float]]
                contrast=(0.5, 1),  # Union[float, Tuple[float, float]]
                saturation=(0.5, 1),  # Union[float, Tuple[float, float]]
                hue=(-0.1, 0.1),  # Union[float, Tuple[float, float]]
                p=1.0,  # float
            )
            # A.Equalize(p=1.0)
    ], p=color)
        
    Ts = []  
    if normalize:
        Ts.append(A.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), 
                    std=(0.26862954, 0.26130258, 0.27577711)))
    if to_tensor:
        Ts.append(ToTensorV2())

    if color > 0:
        Ts = [T_color] + Ts 

    if spatial > 0:
        Ts = [T_spatial] + Ts

    if max(vflip, hflip) > 0:
        Ts = [T_flip] + Ts

    if mask > 0:
        Ts = [T_mask] + Ts
    

    if target_size is not None:
        if isinstance(target_size, (list, tuple)):
            H, W = target_size
        else:
            H = W = target_size
        Ts = [A.Resize(H, W),] + Ts

    if need_skeleton:
        return A.Compose(Ts, 
            keypoint_params=A.KeypointParams(
            format='xy', remove_invisible=remove_invisible, label_fields=['keypoint_data', ]))
    else:
        return A.Compose(Ts)





def denormalize_bchw_image(x):
    pixel_std = torch.as_tensor((0.26862954, 0.26130258, 0.27577711)) * 255
    pixel_mean = torch.as_tensor((0.48145466, 0.4578275, 0.40821073)) * 255

    pixel_std = pixel_std.to(x.device)
    pixel_mean = pixel_mean.to(x.device)
    
    return (x * pixel_std[None, :, None, None]) + pixel_mean[None, :, None, None]