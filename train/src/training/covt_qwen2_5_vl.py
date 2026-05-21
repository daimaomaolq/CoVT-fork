import os
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import torchvision.transforms as T
from torchvision import transforms

from transformers import (
    AutoModel,
    AutoProcessor,
    CLIPImageProcessor,
    AutoImageProcessor,
)
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)

from training.configuration_qwen2_5_vl import (
    Qwen2_5_VLConfig,
    Qwen2_5_VLVisionConfig,
)
from training.modeling_qwen2_5_vl import *
try:
    from diffusers import AutoencoderKL
except Exception:
    AutoencoderKL = None

from anchors.segment_anything import (
    build_sam_vit_h,
    sam_model_registry,
    SamPredictor,
    SamAutomaticMaskGenerator,
)
from anchors.segment_anything.utils.transforms import ResizeLongestSide
from src.anchors.DepthAnything.depth_anything_v2.dpt import DepthAnythingV2
from anchors.pidinet.models import pidinet, pidinet_converted
import anchors.pidinet.models as pidinet_model
from anchors.pidinet.models.config import (
    config_model,
    config_model_converted,
)
from anchors.pidinet.models.convert_pidinet import convert_pidinet
import wandb

def save_seg_image(img, mask, save_path, point=None):
    # img = Image.open(image_path).convert('RGB')
    img = img.resize((256, 256))
    
    mask = (mask > 0).float()
    mask = mask.permute(1, 2, 0).cpu().numpy()
    mask = (mask * 255).astype('uint8')
    mask_img = Image.fromarray(mask.squeeze(), mode='L').convert('RGB')

    img_blend = Image.blend(img, mask_img, alpha=0.5)

    if point is not None:
        draw = ImageDraw.Draw(img_blend)
        r = 7
        x, y = int(point[0]), int(point[1])
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 0, 0))

    img_blend.save(save_path)


def encode_image(image, predictor):
    # image = Image.open(img_path).convert('RGB')
    image = image.resize((256, 256))
    image_np = np.array(image)
    predictor.set_image(image_np)
    features = predictor.get_image_embedding().detach()
    return features

# dice_loss and sigmoid_ce_loss functions are copied from LISA (https://github.com/dvlab-research/LISA)
def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,  # 100000.0,
    eps=1e-6,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss

def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    alpha: float = 0.25,
    gamma: float = 2.0,
    eps: float = 1e-6,
):
    """
    Compute the sigmoid focal loss between `inputs` and the ground truth `targets`.

    Args:
        inputs: A float tensor of arbitrary shape. Raw logits.
        targets: A float tensor with the same shape. Binary labels (0 or 1).
        num_masks: Normalization factor (usually number of masks or positive samples).
        alpha: Focal loss alpha weighting factor.
        gamma: Focal loss focusing parameter.
        eps: Numerical stability.

    Returns:
        Scaled focal loss normalized by num_masks.
    """
    prob = inputs.sigmoid()
    prob = torch.clamp(prob, min=eps, max=1.0 - eps)

    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    modulating_factor = (1 - p_t) ** gamma
    alpha_factor = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_factor * modulating_factor * ce_loss
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss


def dice_coeff(p, g, eps=1e-6):
    """
    Compute Dice coefficient between predictions and ground truth.
    
    Args:
        p: Predicted masks (logits)
        g: Ground truth masks
        eps: Small value for numerical stability
        
    Returns:
        Dice coefficient
    """
    p = p.sigmoid()
    inter = (p * g).sum((-2, -1))
    union = p.sum((-2, -1)) + g.sum((-2, -1))
    return (2*inter + eps) / (union + eps)


def hungarian_matching(cost):
    """
    Perform Hungarian matching between predictions and ground truth.
    
    Args:
        cost: Cost matrix of shape [n_pred, n_gt]
        
    Returns:
        Tuple of (row_indices, col_indices) for optimal assignment
    """
    r, c = linear_sum_assignment(cost.cpu().numpy())
    return torch.as_tensor(r, device=cost.device), torch.as_tensor(c, device=cost.device)


class AnchorLoss():
    def __init__(self, anchor_loss_weight):
        self.anchor_loss_weight = anchor_loss_weight
        (
            self.sam_loss_weight, 
            self.dino_loss_weight, 
            self.depth_loss_weight, 
            self.SD_loss_weight, 
            self.internvit_loss_weight,
            self.pidinet_loss_weight,
            self.siglip_loss_weight,
            self.metaclip_loss_weight,
        ) = (
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        )
        self.loss_fn = nn.MSELoss()
        
    def get_sam_feature_align_loss(self, sam_embed, gt_embed):
        if self.sam_loss_weight > 0:
            sam_loss = self.loss_fn(sam_embed, gt_embed)
            sam_loss = sam_loss * self.sam_loss_weight * 80.
        else:
            sam_loss = 0.
        return sam_loss

    def get_sam_loss(self, sam_masks, gt_masks, num_masks=1):
        if self.sam_loss_weight > 0:
            d_loss = dice_loss(sam_masks, gt_masks, num_masks=num_masks)
            # bce_loss = sigmoid_ce_loss(sam_masks, gt_masks, num_masks=num_masks)
            focal_loss = sigmoid_focal_loss(sam_masks, gt_masks, num_masks=num_masks)
            sam_loss = 0.5 * d_loss + 2.0 * focal_loss
            sam_loss = sam_loss * self.sam_loss_weight
        else:
            sam_loss = 0.
        return sam_loss
    
    def get_sam_token_loss(self, pred_masks, gt_masks, num_masks=1):
        """
        Compute SAM loss using learnable tokens with Hungarian matching.
        
        Args:
            pred_masks: Predicted masks from tokens [n_tokens, H, W]
            gt_masks: Ground truth masks [n_gt, H, W]
            num_masks: Number of ground truth masks
            
        Returns:
            Total loss
        """
        if self.sam_loss_weight <= 0 or num_masks == 0:
            return 0.0
            
        # Compute cost matrix using 1 - dice_coefficient
        with torch.no_grad():
            cost = 1. - dice_coeff(pred_masks.unsqueeze(1), gt_masks.unsqueeze(0))  # [n_tokens, n_gt]
        
        # Perform Hungarian matching
        r, c = hungarian_matching(cost)
        
        # Compute loss for matched pairs
        if len(r) > 0:
            pos_p = pred_masks[r]  # Matched predictions
            pos_g = gt_masks[c]    # Matched ground truth
            
            # Dice loss + BCE loss for matched pairs
            d_loss = 1. - dice_coeff(pos_p, pos_g).mean()
            bce_loss = F.binary_cross_entropy_with_logits(pos_p, pos_g)
            # focal_loss = sigmoid_focal_loss(pos_p, pos_g, num_masks=1)
            loss_pos = d_loss + bce_loss
            
            # Loss for unmatched tokens (background)
            unmatched = torch.tensor([i for i in range(pred_masks.shape[0]) if i not in r.tolist()], 
                                   device=pred_masks.device)
            if len(unmatched) > 0:
                null_gt = torch.zeros_like(pred_masks[unmatched])
                loss_neg = F.binary_cross_entropy_with_logits(pred_masks[unmatched], null_gt)
                loss = loss_pos + 0.1 * loss_neg
            else:
                loss = loss_pos
        else:
            # No matches, all tokens should predict background
            null_gt = torch.zeros_like(pred_masks)
            loss = F.binary_cross_entropy_with_logits(pred_masks, null_gt)
        
        return loss * self.sam_loss_weight
    
    def get_dino_loss(self, dino_embed, gt_embed):
        if self.dino_loss_weight > 0:
            dino_loss = self.loss_fn(dino_embed, gt_embed)
            dino_loss = dino_loss * self.dino_loss_weight * 0.5
        else:
            dino_loss = 0.
        return dino_loss
    
    def get_depth_loss(self, depth, gt_depth):
        if self.depth_loss_weight > 0:
            depth_loss = self.loss_fn(depth, gt_depth)
            depth_loss = depth_loss * self.depth_loss_weight
        else:
            depth_loss = 0.
        return depth_loss
    
    def get_depth_reconstruction_loss(self, pred_depth, gt_depth):
        """
        Args:
            pred_depth: Predicted depth map [B, 1, H, W]
            gt_depth: Ground truth depth map [B, 1, H, W]
            
        Returns:
            Depth reconstruction loss
        """
        if self.depth_loss_weight > 0:
            depth_loss = F.l1_loss(pred_depth, gt_depth)
            depth_loss = depth_loss * self.depth_loss_weight * 0.005
        else:
            depth_loss = 0.
        return depth_loss
    
    def get_SD_loss(self, pred_noise, gt_noise):
        if self.SD_loss_weight > 0:
            SD_loss = self.loss_fn(pred_noise, gt_noise)
            SD_loss = SD_loss * self.SD_loss_weight
        else:
            SD_loss = 0.
        return SD_loss
    
    def get_internvit_loss(self, internvit_embed, gt_embed):
        if self.internvit_loss_weight > 0:
            internvit_loss = self.loss_fn(internvit_embed, gt_embed)
            internvit_loss = internvit_loss * self.internvit_loss_weight
        else:
            internvit_loss = 0.
        return internvit_loss
    
    def get_SD_vae_align_loss(self, pred_latent, gt_latent):
        """
        Qwen SD tokens和VAE teacher latent对齐loss
        pred_latent, gt_latent: [B, 4, 64, 64]
        """
        if self.SD_loss_weight > 0:
            loss = F.mse_loss(pred_latent, gt_latent) * self.SD_loss_weight
        else:
            loss = 0.
        return loss
    
    def get_SD_image_align_loss(self, pred_image, gt_image):
        """
        Qwen SD tokens解码后的图像和teacher图像对齐loss
        pred_image, gt_image: [B, 3, H, W] 归一化到[-1, 1]的图像
        """
        if self.SD_loss_weight > 0:
            l1_loss = F.l1_loss(pred_image, gt_image)
            mse_loss = F.mse_loss(pred_image, gt_image)
            loss = (0.5 * l1_loss + 0.5 * mse_loss) * self.SD_loss_weight
        else:
            loss = 0.
        return loss
    
    def get_pidinet_loss(self, pred_edge, gt_edge):
        if self.pidinet_loss_weight > 0:
            pidinet_loss = F.l1_loss(pred_edge, gt_edge)
            pidinet_loss = pidinet_loss * self.pidinet_loss_weight
        else:
            pidinet_loss = 0.
        return pidinet_loss
    
    def get_siglip_loss(self, pred_slip_embed, gt_slip_embed):
        if self.siglip_loss_weight > 0:
            siglip_loss = self.loss_fn(pred_slip_embed, gt_slip_embed)
            siglip_loss = siglip_loss * self.siglip_loss_weight * 5.0
        else:
            siglip_loss = 0.
        return siglip_loss
    
    def get_metaclip_loss(self, pred_metaclip_embed, gt_metaclip_embed):
        if self.metaclip_loss_weight > 0:
            metaclip_loss = self.loss_fn(pred_metaclip_embed, gt_metaclip_embed)
            metaclip_loss = metaclip_loss * self.metaclip_loss_weight * 5.0
        else:
            metaclip_loss = 0.
        return metaclip_loss


class AnchorModels():
    def __init__(self, anchor_model_id):
        self.anchor_model_id = anchor_model_id        
        SAM_CHECKPOINT = "src/anchors/segment_anything/ckpt/sam_vit_h_4b8939.pth"
        SAM_MODEL_TYPE = "vit_h"
        
        DINO_MODEL_TYPE = "dinov2_vitl14"
        DINO_MODEL_PATH = "facebookresearch/dinov2"
        DINO_PROCESSOR_CONFIG = {"pretrained_model_name_or_path": "facebook/dinov2-large", "crop_size": {"height": 448, "width": 448}}

        DEPTH_MODEL_CONFIG = {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
        DEPTH_CHECKPOINT = "src/anchors/DepthAnything/ckpt/depth_anything_v2_vitl.pth"
        
        SD_MODEL_PATH = "stabilityai/stable-diffusion-2-1-base"
        
        INTERNVIT_MODEL_PATH = "OpenGVLab/InternViT-300M-448px-V2_5"
        
        PIDINET_MODEL_PATH = "src/anchors/pidinet/ckpt/table5_baseline.pth"
        
        SIGLIP_MODEL_PATH = "google/siglip2-large-patch16-256"
        
        METACLIP_MODEL_PATH = "facebook/metaclip-h14-fullcc2.5b"
        
        if "sam" in self.anchor_model_id:
            self.sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
            self.sam.eval()
            self.sam_predictor = SamPredictor(self.sam)
            self.mask_generator = SamAutomaticMaskGenerator(self.sam)
        if "dino" in self.anchor_model_id:
            self.dinovit = torch.hub.load(DINO_MODEL_PATH, DINO_MODEL_TYPE)
            self.dinovit = self.dinovit.eval()
            self.extracted_outputs = {}
            def norm_hook(module, module_input, module_output):
                self.extracted_outputs["norm_output"] = module_output
            self.hook_handle = self.dinovit.norm.register_forward_hook(norm_hook)
            self.dino_processor = AutoImageProcessor.from_pretrained(**DINO_PROCESSOR_CONFIG)
            self.tat_loss = TaTDistillLoss(student_dim=1024, teacher_dim=1024, patch_group=True, group_size=8)
        if "depth" in self.anchor_model_id:
            self.depth_model = DepthAnythingV2(**DEPTH_MODEL_CONFIG)
            self.depth_model.load_state_dict(torch.load(DEPTH_CHECKPOINT, map_location='cpu'))
            self.depth_model = self.depth_model.eval()
            self.depth_layer_idx = [4, 11, 17, 23] 
        if "SD" in self.anchor_model_id:
            self.vae = AutoencoderKL.from_pretrained(
                SD_MODEL_PATH,
                subfolder="vae",
            ).eval()
        if "internvit" in self.anchor_model_id:
            self.internvit = AutoModel.from_pretrained(
                INTERNVIT_MODEL_PATH,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True).eval()
            self.internvit_processor = AutoImageProcessor.from_pretrained(INTERNVIT_MODEL_PATH)
            
        if "pidinet" in self.anchor_model_id:
            class Args:
                def __init__(self):
                    self.model = "pidinet_converted"
                    self.config = "carv4"
                    self.sa = True
                    self.dil = True
            args = Args()
            self.pidinet = pidinet_model.pidinet_converted(args)
            # self.pidinet = torch.nn.DataParallel(self.pidinet).cuda()
            if PIDINET_MODEL_PATH is not None:
                pidinet_checkpoint = torch.load(PIDINET_MODEL_PATH, map_location='cpu')
                pidonet_state_dict = pidinet_checkpoint['state_dict'] if 'state_dict' in pidinet_checkpoint else pidinet_checkpoint
                pidonet_state_dict = {k.replace('module.', ''): v for k, v in pidonet_state_dict.items()}
                self.pidinet.load_state_dict(convert_pidinet(pidonet_state_dict, args.config))
            self.pidinet.eval()
            self.pidi_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])
            
        if "siglip" in self.anchor_model_id:
            self.siglip = AutoModel.from_pretrained(SIGLIP_MODEL_PATH).eval()
            self.siglip_processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_PATH)
            # self.siglip.eval()
            
        if "metaclip" in self.anchor_model_id:
            self.metaclip = AutoModel.from_pretrained(METACLIP_MODEL_PATH).eval()
            self.metaclip_processor = AutoProcessor.from_pretrained(METACLIP_MODEL_PATH)
    
    def set_device(self, device):
        self.device = device
        if "sam" in self.anchor_model_id:
            self.sam.to(device)
        if "dino" in self.anchor_model_id:
            self.dinovit.to(device)
        if "depth" in self.anchor_model_id:
            self.depth_model.to(device)
        if "SD" in self.anchor_model_id:
            self.vae.to(device)
        if "internvit" in self.anchor_model_id:
            self.internvit.to(device)
        if "pidinet" in self.anchor_model_id:
            self.pidinet.to(device)
        if "siglip" in self.anchor_model_id:
            self.siglip.to(device)
        if "metaclip" in self.anchor_model_id:
            self.metaclip.to(device)
    
    def set_float(self):
        if "sam" in self.anchor_model_id:
            self.sam.float()
        if "dino" in self.anchor_model_id:
            self.dinovit.float()
        if "depth" in self.anchor_model_id:
            self.depth_model.float()
        if "SD" in self.anchor_model_id:
            self.vae.float()
        if "internvit" in self.anchor_model_id:
            self.internvit.to(torch.bfloat16)
        if "pidinet" in self.anchor_model_id:
            self.pidinet.float()
        if "siglip" in self.anchor_model_id:
            self.siglip.float()
        if "metaclip" in self.anchor_model_id:
            self.metaclip.float()
        
    def get_sam_embed(self, image_path):
        if "sam" in self.anchor_model_id:
            return encode_image(image_path, self.sam_predictor)  # [1, 256, 64, 64]
        else:
            return None
        
    def decode_sam_embed(self, sam_embed, image, point, label, gt=False):
        # image = Image.open(image_path).convert('RGB')
        image = image.resize((256, 256))
        image_np = np.array(image)
        original_h, original_w = image_np.shape[:2]
        
        transform = ResizeLongestSide(self.sam.image_encoder.img_size)
        input_size = transform.apply_image(image_np).shape[:2]
        
        with torch.no_grad():
            sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                points=(point, label),
                boxes=None,
                masks=None,
            )
        with torch.no_grad():
            low_res_masks, iou_predictions = self.sam.mask_decoder(
                image_embeddings=sam_embed,
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
        upscaled_mask = self.sam.postprocess_masks(
            low_res_masks,
            input_size=input_size,
            original_size=(original_h, original_w)
        )[0]
        if gt:
            upscaled_mask = (upscaled_mask > 0).float()
        return upscaled_mask
    
    def get_sam_mask(self, image):
        # image = Image.open(image_path).convert('RGB')
        image = image.resize((256, 256))
        image_np = np.array(image)
        
        masks = self.mask_generator.generate(image_np)
        masks = sorted(masks, key=lambda x: x["area"], reverse=True)[:4]
        # stability_score = [mask["stability_score"] for mask in masks]
        masks = [mask["segmentation"] for mask in masks if mask["stability_score"] > 0.95]
        num_masks = len(masks)
        masks = torch.tensor(masks, dtype=torch.float, device=self.sam.device)
        return masks, num_masks
    
    def get_point_from_mask(self, image, mask):
        # image = Image.open(image_path).convert('RGB')
        image = image.resize((256, 256))
        image_np = np.array(image)
        original_h, original_w = image_np.shape[:2]
        
        transform = ResizeLongestSide(self.sam.image_encoder.img_size)

        mask_np = mask.cpu().numpy().astype('uint8')
        ys, xs = mask_np.nonzero()
        
        if len(xs) == 0:
            h, w = mask_np.shape
            return [w // 2, h // 2]
        
        idx = torch.randint(0, len(xs), (1,)).item()
        point_xy = [xs[idx], ys[idx]]
        pt_np = np.array([point_xy])
        pt_input = transform.apply_coords(pt_np, (original_h, original_w))[0]
        return pt_input.tolist(), point_xy
    
    def get_point_from_img(self, image):
        # create a all one mask and use get_point_from_mask to get the point
        mask = torch.ones((256, 256), dtype=torch.float, device=self.sam.device)
        # mask = torch.ones((image.size[1], image.size[0]), dtype=torch.float, device=self.sam.device)
        return self.get_point_from_mask(image, mask)
    
    def decode_sam_embed_with_tokens(self, sam_embed, image, token_embeddings):
        """
        Decode SAM masks using learnable token embeddings.
        
        Args:
            sam_embed: SAM image embeddings [1, 256, 64, 64]
            image: PIL Image
            token_embeddings: Learnable token embeddings [n_tokens, 256]
            
        Returns:
            Predicted masks [n_tokens, H, W]
        """
        image = image.resize((256, 256))
        image_np = np.array(image)
        original_h, original_w = image_np.shape[:2]
        
        transform = ResizeLongestSide(self.sam.image_encoder.img_size)
        input_size = transform.apply_image(image_np).shape[:2]
        
        preds = []
        for token_embed in token_embeddings:
            # (1,1,256) -> PromptEncoder
            text_embeds = token_embed.unsqueeze(0).unsqueeze(0)  # B=1,N=1,256
            sparse, dense = self.sam.prompt_encoder(
                points=None, boxes=None, masks=None,
                text_embeds=text_embeds
            )
            low_res_masks, _ = self.sam.mask_decoder(
                image_embeddings=sam_embed,
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=False,
            )  # [1,1,256,256]
            up = self.sam.postprocess_masks(
                low_res_masks,
                input_size=input_size,
                original_size=(original_h, original_w),
            )[0]  # [1,H,W]
            preds.append(up.squeeze(0))
        
        return torch.stack(preds, 0)  # [n_tokens,H,W]
    
    def filter_masks_by_area(self, masks, min_percentile=20, max_percentile=80):
        areas = [mask["area"] for mask in masks]
        areas = np.array(areas)
        min_area = np.percentile(areas, min_percentile)
        max_area = np.percentile(areas, max_percentile)
        filtered_masks = []
        for i, mask in enumerate(masks):
            if min_area <= areas[i] <= max_area:
                filtered_masks.append(mask)
        return filtered_masks
        
    def get_sam_mask_improved(self, image):
        """
        Get SAM masks with improved filtering based on predicted_iou and stability_score.
        
        Args:
            image: PIL Image
            
        Returns:
            masks: Tensor of masks [num_masks, H, W]
            num_masks: Number of valid masks
        """
        image = image.resize((256, 256))
        image_np = np.array(image)
        
        masks = self.mask_generator.generate(image_np)
        # masks = self.filter_masks_by_area(masks, min_percentile=20, max_percentile=80)
        # Sort by predicted_iou * stability_score and take top 8
        masks = sorted(masks, key=lambda x: (x["predicted_iou"] * x["stability_score"]), reverse=True)[:8]
        # Filter by stability_score > 0.95
        masks = [mask["segmentation"].astype(np.float32) for mask in masks]
        num_masks = len(masks)
        print(f'the number of masks: {num_masks}')
        
        if num_masks > 0:
            masks = torch.tensor(masks, dtype=torch.float, device=self.sam.device)
        else:
            masks = torch.empty((0, image_np.shape[0], image_np.shape[1]), dtype=torch.float, device=self.sam.device)
            
        return masks, num_masks
    
    def get_dino_embed(self, image, device):
        if "dino" in self.anchor_model_id:
            # image = Image.open(image_path).convert("RGB")
            dino_pixel_value = self.dino_processor(images=image, return_tensors='pt')['pixel_values']
            dino_pixel_value = dino_pixel_value.to(device)
            dino_val = self.dinovit(dino_pixel_value)
            dino_features = self.extracted_outputs["norm_output"].detach()
            del dino_val
            torch.cuda.empty_cache()
            return dino_features  # [1, 1025, 1024]
        else:
            return None
        
    def get_depth_embed(self, raw_img):
        if "depth" in self.anchor_model_id:
            raw_img = raw_img.resize((256, 256))
            raw_img = np.array(raw_img)
            img, _ = self.depth_model.image2tensor(raw_img)
            feature = self.depth_model.pretrained.get_intermediate_layers(img, [23])
            return feature[0].detach()  # [1, 1369, 1024]
        else:
            return None
    
    def get_depth_features_and_gt(self, raw_img):
        """
        Args:
            raw_img: PIL Image
            
        Returns:
            patch_feats: List of patch features from different layers [4][B, N, C]
            cls_token: CLS token [B, C]
            depth_gt: Ground truth depth map [B, 1, H, W]
            patch_hw: Patch height and width (Hf, Wf)
            img_hw: Original image height and width (H, W)
        """
        if "depth" not in self.anchor_model_id:
            return None, None, None, None, None
            
        raw_img = raw_img.resize((256, 256))
        raw_img = np.array(raw_img)
        img, (H_raw, W_raw) = self.depth_model.image2tensor(raw_img)
        img = img.to(self.device)
        
        patch_h, patch_w = img.shape[-2] // 14, img.shape[-1] // 14
        
        feats_raw = self.depth_model.pretrained.get_intermediate_layers(
            img, self.depth_layer_idx, return_class_token=True
        )
        patch_feats = [f[0] for f in feats_raw]  # patches [B, N, C]
        cls_token = feats_raw[-1][1].squeeze(1)  # CLS [B, C]
        
        with torch.no_grad():
            depth_gt = self.depth_model.depth_head(feats_raw, patch_h, patch_w)  # [B, 1, Hf, Wf]
            depth_gt = F.relu(depth_gt)
            depth_gt = F.interpolate(depth_gt, size=(H_raw, W_raw),
                                   mode='bilinear', align_corners=True)
        
        return patch_feats, cls_token, depth_gt, (patch_h, patch_w), (H_raw, W_raw)
        
    def decode_depth_embed(self, depth_embed):
        if "depth" in self.anchor_model_id:
            depth_embed = depth_embed.view(1, 1369, 1024)
            depth_embed = self.depth_model.pretrained.decode_head(depth_embed)
            return depth_embed
        else:
            return None
        
    def get_SD_embed(self, img):
        if "SD" in self.anchor_model_id:
            read_image = transforms.ToTensor()(img).unsqueeze(0).to('cuda')
            read_image = torch.nn.functional.interpolate(
                read_image, size=(512, 512), mode='bilinear', align_corners=False
            )
            img = 2 * read_image - 1
            feature = self.vae.encode(img).latent_dist.sample() * 0.18215
            del img
            return feature.detach()  # [1, 4, 64, 64]
        else:
            return None
        
    def get_internvit_embed(self, raw_img):
        if "internvit" in self.anchor_model_id:
            pixel_values = self.internvit_processor(images=raw_img, return_tensors='pt').pixel_values
            pixel_values = pixel_values.to(torch.bfloat16).cuda()
            outputs = self.internvit(pixel_values)
            return outputs.last_hidden_state.detach()  # [1, 1025, 1024]
        else:
            return None
        
    def get_SD_teacher_latent(self, img):
        if "SD" in self.anchor_model_id:
            read_image = transforms.ToTensor()(img).unsqueeze(0).to(self.device)
            read_image = torch.nn.functional.interpolate(
                read_image, size=(512, 512), mode='bilinear', align_corners=False
            )
            img = 2 * read_image - 1
            feature = self.vae.encode(img).latent_dist.sample() * 0.18215
            return feature.detach()
        else:
            return None
        
    def get_dino_tat_loss(self, stu_feat, tch_feat):
        return self.tat_loss(stu_feat, tch_feat) * 5e3

    def get_pidinet_embed(self, image):
        if "pidinet" in self.anchor_model_id:
            image = image.resize((256, 256))
            if not hasattr(self, "pidinet"):
                return None
            x = self.pidi_normalize(T.ToTensor()(image)).unsqueeze(0)
            x = x.to(self.pidinet.parameters().__next__().device)
            with torch.no_grad():
                _, out, _ = self.pidinet(x)
            return out.squeeze().unsqueeze(0).detach()
        else:
            return None
        
    def get_pidinet_mid_feats(self, image):
        if "pidinet" in self.anchor_model_id:
            image = image.resize((256, 256))
            if not hasattr(self, "pidinet"):
                return None
            x = self.pidi_normalize(T.ToTensor()(image)).unsqueeze(0)
            x = x.to(self.pidinet.parameters().__next__().device)
            with torch.no_grad():
                _, _, mid_feats = self.pidinet(x)
            return mid_feats.detach()
        else:
            return None
        
    def get_edge_from_tokens(self, tokens, mid_feats):
        batch_size, token_dim = tokens.shape[0], tokens.shape[1]
        conv_weights = tokens.view(batch_size, token_dim, 60, 1, 1).float()
        aligned_features_list = []
        for b in range(batch_size):
            curr_weights = conv_weights[b]  # [4, 60, 1, 1]
            curr_x1 = mid_feats[b:b+1]  # [1, 60, H, W]
            curr_features = []
            for t in range(token_dim):
                weight = curr_weights[t:t+1]  # [1, 60, 1, 1][]
                feature = F.conv2d(curr_x1, weight, bias=None)  # [1, 1, H, W]
                curr_features.append(feature)
            curr_aligned = torch.cat(curr_features, dim=1)
            aligned_features_list.append(curr_aligned)
        aligned_features = torch.cat(aligned_features_list, dim=0)
        predicted_edge = torch.mean(aligned_features, dim=1, keepdim=True)
        predicted_edge = torch.sigmoid(predicted_edge)
        return predicted_edge
                
    def get_siglip_embed(self, image):
        if "siglip" in self.anchor_model_id:
            inputs = self.siglip_processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                image_embeddings = self.siglip.get_image_features(**inputs)
            return image_embeddings.detach()
        else:
            return None
    
    def get_metaclip_embed(self, image):
        if "metaclip" in self.anchor_model_id:
            inputs = self.metaclip_processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                image_embeddings = self.metaclip.get_image_features(**inputs)
            return image_embeddings.detach()
        else:
            return None


class TaTDistillLoss(nn.Module):
    def __init__(self, student_dim, teacher_dim, patch_group=True, group_size=8):
        super().__init__()
        self.patch_group = patch_group
        self.group_size = group_size
        self.proj_s = nn.Linear(student_dim, teacher_dim) if student_dim != teacher_dim else nn.Identity()
        self.proj_t = nn.Identity()
    def forward(self, feat_s, feat_t):
        # feat_s, feat_t: [B, N, C]
        if self.patch_group:
            B, N, C = feat_s.shape
            G = N // self.group_size
            feat_s = feat_s[:, 1:, :].reshape(B, G, self.group_size, C).reshape(B, G, -1)
            feat_t = feat_t[:, 1:, :].reshape(B, G, self.group_size, C).reshape(B, G, -1)
        else:
            feat_s = feat_s[:, 1:, :]
            feat_t = feat_t[:, 1:, :]
        feat_s = self.proj_s(feat_s)
        feat_t = self.proj_t(feat_t)
        B, N, C = feat_s.shape
        loss = 0.
        for b in range(B):
            s = F.normalize(feat_s[b], dim=-1)
            t = F.normalize(feat_t[b], dim=-1)
            sim = torch.matmul(s, t.T)
            attn = F.softmax(sim, dim=0)
            s_recfg = torch.matmul(attn.T, s)
            loss += F.mse_loss(s_recfg, t)
        return loss / B


class CoVTForConditionalGeneration(Qwen2_5_VLPreTrainedModel, GenerationMixin):
        
    _tied_weights_keys = ["lm_head.weight"]
    config_class = Qwen2_5_VLConfig
    _no_split_modules = ["Qwen2_5_VLDecoderLayer", "Qwen2_5_VLVisionBlock"]

    def __init__(self, config):
        
        super().__init__(config)
        self.visual = Qwen2_5_VisionTransformerPretrainedModel._from_config(config.vision_config)
        self.model = Qwen2_5_VLModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.rope_deltas = None  # cache rope_deltas here
        
        self.anchor_model_id = None
        self.anchor_loss_weight = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # No Use
        self.anchor_models = None
        self.anchor_loss = None
        
        self.sam_projection = None
        self.dino_projection = None
        self.depth_projection = None
        self.SD_projection = None
        self.internvit_projection = None
        self.pidinet_projection = None
        self.siglip_projection = None
        self.metaclip_projection = None
        
        self.sam_query_vectors = None
        self.dino_query_vectors = None
        self.depth_query_vectors = None
        self.SD_query_vectors = None
        self.internvit_query_vectors = None
        self.pidinet_query_vectors = None
        self.siglip_query_vectors = None
        self.metaclip_query_vectors = None

        self.sam_cross_attention = None
        self.dino_cross_attention = None
        self.depth_cross_attention = None
        self.SD_cross_attention = None
        self.internvit_cross_attention = None
        self.pidinet_cross_attention = None
        self.siglip_cross_attention = None
        self.metaclip_cross_attention = None
        
        self.depth_token_generator = None
        
        class DepthReconstructor(nn.Module):
            def __init__(self):
                super().__init__()
                
            def forward(self, tokens, patch_feats, patch_hw, img_hw):
                """
                tokens[B,4,C] + patch_feats(list[4][B,N,C]) → (tokens_depth, depth_avg)
                """
                B, T, C = tokens.shape
                Hf, Wf = patch_hw
                outs = []
                for i in range(T):
                    tok = tokens[:, i, :].unsqueeze(1)          # [B,1,C]
                    f = patch_feats[i]                          # [B,N,C]
                    tok = tok.to(f.dtype)
                    score = torch.bmm(tok, f.transpose(1, 2))   # [B,1,N]
                    score = score.squeeze(1).view(B, 1, Hf, Wf) # → [B,1,Hf,Wf]
                    up = F.interpolate(score, size=img_hw, mode='bilinear', align_corners=False)
                    outs.append(up.squeeze(1))                  # [B,H,W]
                token_depths = torch.stack(outs, dim=1)         # [B,4,H,W]
                depth_avg = token_depths.mean(1, keepdim=True)
                return token_depths, depth_avg
        
        self.depth_reconstructor = DepthReconstructor()
        
        # global step for deciding loss strategy
        self.global_steps = 0
        self.align_feature_stage = 0
        self.align_anchor_task_only_stage = 0
        self.align_vqa_only_stage = 6000
        
        # Initialize weights and apply final processing
        self.post_init()
        
        self.sam_token_idx = None
        self.dino_token_idx = None
        self.depth_token_idx = None
        self.SD_token_idx = None
        self.internvit_token_idx = None
        self.pidinet_token_idx = None
        self.siglip_token_idx = None
        self.metaclip_token_idx = None
        
        self.sam_projection = nn.Linear(3584, 256)
        self.sam_query_vectors = nn.Parameter(torch.randn(8, 256, dtype=torch.bfloat16, requires_grad=True))
        self.sam_cross_attention = nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)
        self.dino_projection = nn.Linear(3584, 1024)
        self.dino_query_vectors = nn.Parameter(torch.randn(1025, 1024, dtype=torch.bfloat16, requires_grad=True))
        self.dino_cross_attention = nn.MultiheadAttention(embed_dim=1024, num_heads=8, batch_first=True)
        self.depth_projection = nn.Linear(3584, 1024)
        self.depth_query_vectors = nn.Parameter(torch.randn(1369, 1024, dtype=torch.bfloat16, requires_grad=True))
        self.depth_cross_attention = nn.MultiheadAttention(embed_dim=1024, num_heads=8, batch_first=True)
        self.depth_token_generator = nn.Sequential(
            nn.Linear(3584, 3584),
            nn.GELU(),
            nn.Linear(3584, 1024)
        )
        self.SD_projection = nn.Linear(3584, 4096)
        self.SD_query_vectors = nn.Parameter(torch.randn(4, 4096, dtype=torch.bfloat16, requires_grad=True))
        self.SD_cross_attention = nn.MultiheadAttention(embed_dim=4096, num_heads=8, batch_first=True)
        self.internvit_projection = nn.Linear(3584, 1024)
        self.internvit_query_vectors = nn.Parameter(torch.randn(1025, 1024, dtype=torch.bfloat16, requires_grad=True))
        self.internvit_cross_attention = nn.MultiheadAttention(embed_dim=1024, num_heads=8, batch_first=True)
        self.pidinet_projection = nn.Linear(3584, 60)
        self.pidinet_query_vectors = nn.Parameter(torch.randn(4, 60, dtype=torch.bfloat16, requires_grad=True))
        self.pidinet_cross_attention = nn.MultiheadAttention(embed_dim=60, num_heads=4, batch_first=True)
        self.siglip_projection = nn.Linear(3584, 1024)
        self.siglip_query_vectors = nn.Parameter(torch.randn(1, 1024, dtype=torch.bfloat16, requires_grad=True))
        self.siglip_cross_attention = nn.MultiheadAttention(embed_dim=1024, num_heads=8, batch_first=True)
        self.metaclip_projection = nn.Linear(3584, 1024)
        self.metaclip_query_vectors = nn.Parameter(torch.randn(1, 1024, dtype=torch.bfloat16, requires_grad=True))
        self.metaclip_cross_attention = nn.MultiheadAttention(embed_dim=1024, num_heads=8, batch_first=True)
        
        # self.SD_token_projection = nn.Linear(3584, 64*64)
        
    def get_anchor_model_ids(self, anchor_model_id):
        self.anchor_model_id = anchor_model_id
        self.anchor_models = AnchorModels(self.anchor_model_id)
        self.anchor_loss = AnchorLoss(self.anchor_loss_weight)
        
        if "sam" not in self.anchor_model_id:
            self.sam_projection = None
            self.sam_query_vectors = None
            self.sam_cross_attention = None
        if "dino" not in self.anchor_model_id:
            self.dino_projection = None
            self.dino_query_vectors = None
            self.dino_cross_attention = None
        if "depth" not in self.anchor_model_id:
            self.depth_projection = None
            self.depth_query_vectors = None
            self.depth_cross_attention = None
            self.depth_token_generator = None
        if "SD" not in self.anchor_model_id:
            self.SD_projection = None
            self.SD_query_vectors = None
            self.SD_cross_attention = None
        if "internvit" not in self.anchor_model_id:
            self.internvit_projection = None
            self.internvit_query_vectors = None
            self.internvit_cross_attention = None
        if "pidinet" not in self.anchor_model_id:
            self.pidinet_projection = None
            self.pidinet_query_vectors = None
            self.pidinet_cross_attention = None
        if "siglip" not in self.anchor_model_id:
            self.siglip_projection = None
            self.siglip_query_vectors = None
            self.siglip_cross_attention = None
        if "metaclip" not in self.anchor_model_id:
            self.metaclip_projection = None
            self.metaclip_query_vectors = None
            self.metaclip_cross_attention = None
        
        
    def get_anchor_token_idx(self, sam_token_idx, dino_token_idx, depth_token_idx, SD_token_idx, internvit_token_idx, pidinet_token_idx, siglip_token_idx, metaclip_token_idx):
        self.sam_token_idx = sam_token_idx
        self.dino_token_idx = dino_token_idx
        self.depth_token_idx = depth_token_idx
        self.SD_token_idx = SD_token_idx
        self.internvit_token_idx = internvit_token_idx
        self.pidinet_token_idx = pidinet_token_idx
        self.siglip_token_idx = siglip_token_idx
        self.metaclip_token_idx = metaclip_token_idx

    def _collect_token_hidden_states(self, input_ids, hidden_states, token_idx):
        if token_idx is None:
            return [], []

        token_mask = input_ids == token_idx
        valid_indices = []
        token_features = []
        for batch_idx in range(input_ids.shape[0]):
            if token_mask[batch_idx].any():
                valid_indices.append(batch_idx)
                token_features.append(hidden_states[batch_idx, token_mask[batch_idx]])

        if not token_features:
            return [], []

        lengths = {features.shape[0] for features in token_features}
        if len(lengths) == 1:
            token_features = torch.stack(token_features, dim=0)

        return token_features, valid_indices

    def extract_visual_thought_tokens(
        self,
        input_ids,
        hidden_states,
        project=True,
        cross_attention=True,
        detach=False,
    ):
        """
        Extract CoVT visual thought token states from a completed forward pass.

        The returned raw states are the hidden states at the special anchor token
        positions. When project=True, the method also returns the same projected
        and optionally cross-attended representations used by the anchor losses.
        """

        def maybe_detach(value):
            if detach and isinstance(value, torch.Tensor):
                return value.detach()
            return value

        outputs = {}
        sam_hidden, sam_valid_indices = self._collect_token_hidden_states(
            input_ids, hidden_states, self.sam_token_idx
        )
        dino_hidden, dino_valid_indices = self._collect_token_hidden_states(
            input_ids, hidden_states, self.dino_token_idx
        )

        outputs["sam"] = {
            "hidden_states": maybe_detach(sam_hidden),
            "valid_indices": sam_valid_indices,
        }
        outputs["dino"] = {
            "hidden_states": maybe_detach(dino_hidden),
            "valid_indices": dino_valid_indices,
        }

        if not project:
            return outputs

        if isinstance(sam_hidden, torch.Tensor) and self.sam_projection is not None:
            sam_projected = self.sam_projection(self.apply_rope_custome(sam_hidden))
            outputs["sam"]["projected"] = maybe_detach(sam_projected)
            if cross_attention and self.sam_cross_attention is not None and self.sam_query_vectors is not None:
                sam_query = self.sam_query_vectors.unsqueeze(0)
                sam_query = sam_query.expand(len(sam_valid_indices), -1, -1).to(sam_projected.dtype)
                sam_proj_norm = nn.functional.normalize(sam_projected)
                sam_attn_output, _ = self.sam_cross_attention(
                    query=sam_query,
                    key=sam_proj_norm,
                    value=sam_proj_norm,
                )
                outputs["sam"]["attended"] = maybe_detach(sam_attn_output.reshape(sam_projected.shape))

        if isinstance(dino_hidden, torch.Tensor) and self.dino_projection is not None:
            dino_projected = nn.functional.normalize(self.dino_projection(dino_hidden))
            outputs["dino"]["projected"] = maybe_detach(dino_projected)
            if cross_attention and self.dino_cross_attention is not None and self.dino_query_vectors is not None:
                dino_query = self.dino_query_vectors.unsqueeze(0)
                dino_query = dino_query.expand(len(dino_valid_indices), -1, -1).to(dino_projected.dtype)
                dino_attn_output, _ = self.dino_cross_attention(
                    query=dino_query,
                    key=dino_projected,
                    value=dino_projected,
                )
                outputs["dino"]["attended"] = maybe_detach(dino_attn_output)

        return outputs

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embeddin for text part.
            Examples:
                Temporal (Time): 3 patches, representing different segments of the video in time.
                Height: 2 patches, dividing each frame vertically.
                Width: 2 patches, dividing each frame horizontally.
                We also have some important parameters:
                fps (Frames Per Second): The video's frame rate, set to 1. This means one frame is processed each second.
                tokens_per_second: This is a crucial parameter. It dictates how many "time-steps" or "temporal tokens" are conceptually packed into a one-second interval of the video. In this case, we have 25 tokens per second. So each second of the video will be represented with 25 separate time points. It essentially defines the temporal granularity.
                temporal_patch_size: The number of frames that compose one temporal patch. Here, it's 2 frames.
                interval: The step size for the temporal position IDs, calculated as tokens_per_second * temporal_patch_size / fps. In this case, 25 * 2 / 1 = 50. This means that each temporal patch will be have a difference of 50 in the temporal position IDs.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [101, 102, 103, 104, 105]
                text height position_ids: [101, 102, 103, 104, 105]
                text width position_ids: [101, 102, 103, 104, 105]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
                The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_index, video_index = 0, 0
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        second_per_grid_t = 0
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image

                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        if second_per_grid_ts is not None:
                            second_per_grid_t = second_per_grid_ts[video_index]
                        else:
                            second_per_grid_t = 1.0
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                    expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)

                    time_tensor = expanded_range * second_per_grid_t * self.config.vision_config.tokens_per_second

                    time_tensor_long = time_tensor.long()
                    t_index = time_tensor_long.flatten()

                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas
        
    def apply_rope_custome(self, x):
        N, K = x.shape[-2], x.shape[-1]
        pad = (K % 2 == 1)
        if pad:
            x = torch.nn.functional.pad(x, (0, 1))
            K += 1
        half = K // 2
        x1, x2 = x[..., :half], x[..., half:]

        idx = torch.arange(half, device=x.device, dtype=x.dtype)
        theta = torch.exp(-torch.log(torch.tensor(10000.0, device=x.device, dtype=x.dtype)) * (2*idx / K))
        pos = torch.arange(N, device=x.device, dtype=x.dtype).unsqueeze(-1)  # [N,1]
        ang = pos * theta  # [N, half]
        cos, sin = torch.cos(ang), torch.sin(ang)

        while cos.dim() < x1.dim():
            cos = cos.unsqueeze(0); sin = sin.unsqueeze(0)

        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        y = torch.cat([y1, y2], dim=-1)
        return y[..., :K - int(pad)]


    @add_start_docstrings_to_model_forward(QWEN2_5_VL_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Qwen2_5_VLCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_files: Optional[List] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
                
        Returns:
        """
                 
        self.global_steps += 1
        
        if self.anchor_models is not None:
            self.anchor_models.set_device(self.device)
            self.anchor_models.set_float()
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)

                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )

                mask = input_ids == self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)

                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts,
                    attention_mask,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
                

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        
        # return dict
        predict_masks_dict = None
        predict_dino_dict = None
        predict_depth_dict = None
        predict_SD_dict = None
        predict_internvit_dict = None
        predict_pidinet_dict = None
        predict_siglip_dict = None
        predict_metaclip_dict = None
        
        # calculate anchor loss
        seg_loss = 0.0
        dino_loss = 0.0
        depth_loss = 0.0
        SD_loss = 0.0
        internvit_loss = 0.0
        pidinet_loss = 0.0
        siglip_loss = 0.0
        metaclip_loss = 0.0
            
        sam_encoded_values = []
        dino_encoded_values = []
        depth_encoded_values = []
        SD_encoded_values = []
        internvit_encoded_values = []
        pidinet_encoded_values = []
        siglip_encoded_values = []
        metaclip_encoded_values = []

        sam_encoded_value = None
        dino_encoded_value = None
        depth_encoded_value = None
        SD_encoded_value = None
        internvit_encoded_value = None
        pidinet_encoded_value = None
        siglip_encoded_value = None
        metaclip_encoded_value = None

        if image_files and self.global_steps <= self.align_vqa_only_stage:
            for image_file in image_files:
                sam_embed = self.anchor_models.get_sam_embed(image_file[0])
                sam_encoded_values.append(sam_embed)
                dino_embed = self.anchor_models.get_dino_embed(image_file[0], self.device)
                dino_encoded_values.append(dino_embed)
                depth_embed = self.anchor_models.get_depth_embed(image_file[0])
                depth_encoded_values.append(depth_embed)
                SD_embed = self.anchor_models.get_SD_embed(image_file[0])
                SD_encoded_values.append(SD_embed)
                internvit_embed = self.anchor_models.get_internvit_embed(image_file[0])
                internvit_encoded_values.append(internvit_embed)
                pidinet_embed = self.anchor_models.get_pidinet_embed(image_file[0])
                pidinet_encoded_values.append(pidinet_embed)
                siglip_embed = self.anchor_models.get_siglip_embed(image_file[0])
                siglip_encoded_values.append(siglip_embed)
                metaclip_embed = self.anchor_models.get_metaclip_embed(image_file[0])
                metaclip_encoded_values.append(metaclip_embed)

            sam_encoded_value = torch.cat(sam_encoded_values, dim=0).to(outputs.hidden_states[-1].dtype) if (sam_encoded_values[0] is not None) else None
            dino_encoded_value = torch.cat(dino_encoded_values, dim=0).to(outputs.hidden_states[-1].dtype) if (dino_encoded_values[0] is not None) else None
            depth_encoded_value = torch.cat(depth_encoded_values, dim=0).to(outputs.hidden_states[-1].dtype) if (depth_encoded_values[0] is not None) else None
            SD_encoded_value = torch.cat(SD_encoded_values, dim=0).to(outputs.hidden_states[-1].dtype) if (SD_encoded_values[0] is not None) else None
            internvit_encoded_value = torch.cat(internvit_encoded_values, dim=0).to(outputs.hidden_states[-1].dtype) if (internvit_encoded_values[0] is not None) else None
            pidinet_encoded_value = torch.cat(pidinet_encoded_values, dim=0).to(outputs.hidden_states[-1].dtype) if (pidinet_encoded_values[0] is not None) else None
            siglip_encoded_value = torch.cat(siglip_encoded_values, dim=0).to(outputs.hidden_states[-1].dtype) if (siglip_encoded_values[0] is not None) else None
            metaclip_encoded_value = torch.cat(metaclip_encoded_values, dim=0).to(outputs.hidden_states[-1].dtype) if (metaclip_encoded_values[0] is not None) else None
        
        if sam_encoded_value is not None:
            sam_id_mask = (input_ids == self.sam_token_idx)
            # check if sam_mask is all False
            if (~sam_id_mask.bool()).all() == True:
                seg_loss = 0.0
            else:
                bs = sam_encoded_value.shape[0]
                last_hidden_state = outputs.hidden_states[-1]
                sam_valid_indices = []
                sam_hidden_features = []
                for i_ in range(bs):
                    if sam_id_mask[i_].any():
                        sam_valid_indices.append(i_)
                        sam_features = last_hidden_state[i_, sam_id_mask[i_]]
                        sam_hidden_features.append(sam_features)
                
                if len(sam_hidden_features) > 0:
                    sam_hidden_features = torch.stack(sam_hidden_features, dim=0)
                
                sam_hidden_features = self.apply_rope_custome(sam_hidden_features)
                
                # Project hidden features to 256 (prompt_encoder.embed_dim)
                sam_token_embeddings = self.sam_projection(sam_hidden_features)  # [B, k, 256]
                sam_query = self.sam_query_vectors.unsqueeze(0)
                sam_query = sam_query.expand(len(sam_valid_indices), -1, -1).to(sam_encoded_value.dtype)
                sam_proj = nn.functional.normalize(sam_token_embeddings)
                sam_attn_output, _ = self.sam_cross_attention(
                    query=sam_query,
                    key=sam_proj,
                    value=sam_proj
                )
                sam_attn_output = sam_attn_output.reshape(sam_token_embeddings.shape)
                # sam_attn_output = sam_attn_output.reshape(sam_encoded_value.shape)
                if self.global_steps <= self.align_feature_stage:
                    pass
                                    
                else:
                    # Use token-based training with Hungarian matching
                    for idx, i in enumerate(sam_valid_indices):
                        if len(image_files[i]) > 0:
                            # Get SAM image embeddings
                            sam_embed = self.anchor_models.get_sam_embed(image_files[i][0])
                            
                            # Get ground truth masks
                            gt_masks, num_masks = self.anchor_models.get_sam_mask_improved(image_files[i][0])
                            
                            # Decode masks using token embeddings
                            token_embeddings = sam_attn_output[idx]  # [k, 256]
                            pred_masks = self.anchor_models.decode_sam_embed_with_tokens(
                                sam_embed, image_files[i][0], token_embeddings
                            )  # [k, H, W]
                            
                            if return_dict:
                                predict_masks_dict = {"image_file": image_files[i][0], 
                                                        "pred_masks": (pred_masks > 0).float(),
                                                        "gt_masks": (gt_masks > 0).float(),
                                                        "num_masks": num_masks}
                            
                            # Compute loss using Hungarian matching
                            seg_loss += self.anchor_loss.get_sam_token_loss(pred_masks, gt_masks, num_masks)
                    seg_loss = seg_loss / len(sam_valid_indices) if len(sam_valid_indices) > 0 else 0.0
            try:
                wandb.log({"seg_loss": seg_loss})
            except:
                pass
            
        if dino_encoded_value is not None:
            dino_query = self.dino_query_vectors.unsqueeze(0)
            bs = dino_encoded_value.shape[0]
            dino_mask = (input_ids == self.dino_token_idx)
            # check if dino_mask is all False
            if (~dino_mask.bool()).all() == True:
                dino_loss = 0.0
            else:
                last_hidden_state = outputs.hidden_states[-1]
                dino_valid_indices = []
                dino_hidden_features = []
                for i_ in range(bs):
                    if dino_mask[i_].any():
                        dino_valid_indices.append(i_)
                        dino_features = last_hidden_state[i_, dino_mask[i_]]
                        dino_hidden_features.append(dino_features)
                
                if len(dino_hidden_features) > 0:
                    dino_hidden_features = torch.stack(dino_hidden_features, dim=0)
                    dino_proj = nn.functional.normalize(self.dino_projection(dino_hidden_features)) # [B , 64 , 1024]
                    dino_query_valid = dino_query.expand(len(dino_valid_indices), -1, -1).to(dino_encoded_value.dtype)
                    dino_attn_output, _ = self.dino_cross_attention(
                        query=dino_query_valid,        # Shape: [valid_batch_size, 1025, 1024]
                        key=dino_proj,        # Shape: [valid_batch_size, 64, 1024]
                        value=dino_proj       # Shape: [valid_batch_size, 64, 1024]
                    )
                    dino_features_aligned = dino_encoded_value[dino_valid_indices].to(dino_hidden_features.dtype)
                    dino_loss = self.anchor_loss.get_dino_loss(dino_attn_output, dino_features_aligned)
                else:
                    dino_loss = 0.0
                if return_dict:
                    predict_dino_dict = {"image_file": image_files[0][0], 
                                          "pred_dino": dino_attn_output,
                                          "gt_dino": dino_features_aligned}
            try:
                wandb.log({"dino_loss": dino_loss})
            except:
                pass
            
        if depth_encoded_value is not None:
            depth_query = self.depth_query_vectors.unsqueeze(0)
            bs = depth_encoded_value.shape[0]
            depth_mask = (input_ids == self.depth_token_idx)
            # check if depth_mask is all False
            if (~depth_mask.bool()).all() == True:
                depth_loss = 0.0
            else:
                last_hidden_state = outputs.hidden_states[-1]
                depth_valid_indices = []
                depth_hidden_features = []
                for i_ in range(bs):
                    if depth_mask[i_].any():
                        depth_valid_indices.append(i_)
                        depth_features = last_hidden_state[i_, depth_mask[i_]]
                        depth_hidden_features.append(depth_features)
                
                if len(depth_hidden_features) > 0:
                    depth_hidden_features = torch.stack(depth_hidden_features, dim=0)
                    
                    if self.global_steps <= self.align_feature_stage:
                        depth_proj = nn.functional.normalize(self.depth_projection(depth_hidden_features)) # [B , 64 , 1024]
                        depth_query_valid = depth_query.expand(len(depth_valid_indices), -1, -1).to(depth_encoded_value.dtype)
                        depth_attn_output, _ = self.depth_cross_attention(
                            query=depth_query_valid,        # Shape: [valid_batch_size, 1813, 1024]
                            key=depth_proj,        # Shape: [valid_batch_size, 64, 1024]
                            value=depth_proj       # Shape: [valid_batch_size, 64, 1024]
                        )
                        depth_features_aligned = depth_encoded_value[depth_valid_indices].to(depth_hidden_features.dtype)
                        # Compute MSE loss
                        depth_loss = self.anchor_loss.get_depth_loss(depth_attn_output, depth_features_aligned)
                    else:
                        depth_loss = 0.0
                        for idx, i in enumerate(depth_valid_indices):
                            if len(image_files[i]) > 0:
                                patch_feats, cls_token, depth_gt, patch_hw, img_hw = self.anchor_models.get_depth_features_and_gt(image_files[i][0])
                                
                                if patch_feats is not None:
                                    depth_tokens = self.depth_token_generator(depth_hidden_features[idx])  # [4*1024]
                                    depth_tokens = depth_tokens.view(4, 1024)  # [4, 1024]
                                    depth_tokens = depth_tokens.unsqueeze(0)  # [1, 4, 1024]
                                    
                                    token_depths, depth_pred = self.depth_reconstructor(
                                        depth_tokens, patch_feats, patch_hw, img_hw
                                    )
                                    
                                    depth_loss += self.anchor_loss.get_depth_reconstruction_loss(depth_pred, depth_gt)
                                    
                                    if return_dict:
                                        predict_depth_dict = {"image_file": image_files[i][0], 
                                                              "pred_depth": depth_pred,
                                                              "gt_depth": depth_gt}
                        depth_loss = depth_loss / len(depth_valid_indices) if len(depth_valid_indices) > 0 else 0.0
                else:
                    depth_loss = 0.0
            try:
                wandb.log({"depth_loss": depth_loss})
            except:
                pass
        else:
            depth_loss = 0.0
        
        if SD_encoded_value is not None:
            SD_query = self.SD_query_vectors.unsqueeze(0)
            bs = SD_encoded_value.shape[0]
            SD_mask = (input_ids == self.SD_token_idx)
            if (~SD_mask.bool()).all() == True:
                SD_vae_loss = 0.0
            else:
                last_hidden_state = outputs.hidden_states[-1]
                SD_valid_indices = []
                SD_hidden_features = []
                for i_ in range(bs):
                    if SD_mask[i_].any():
                        SD_valid_indices.append(i_)
                        SD_features = last_hidden_state[i_, SD_mask[i_]]
                        SD_hidden_features.append(SD_features)
                
                if len(SD_hidden_features) > 0:
                    SD_hidden_features = torch.stack(SD_hidden_features, dim=0) # [B, 4, hidden_dim]
                    
                    SD_proj = nn.functional.normalize(self.SD_projection(SD_hidden_features)) # [B, 4, SD_hiddensize]
                    SD_query_valid = SD_query.expand(len(SD_valid_indices), -1, -1).to(SD_encoded_value.dtype)
                    SD_attn_output, _ = self.SD_cross_attention(
                        query=SD_query_valid,        # Shape: [valid_batch_size, 4*64*64, SD_hiddensize]
                        key=SD_proj,           # Shape: [valid_batch_size, 4, SD_hiddensize]
                        value=SD_proj          # Shape: [valid_batch_size, 4, SD_hiddensize]
                    )
                    
                    SD_pred_latent = SD_attn_output # [B, 4*64*64, 4*64*64]
                    SD_pred_latent = SD_pred_latent.view(len(SD_valid_indices), 4, 64, 64)  # [B, 4, 64, 64]
                    
                    SD_pred_image = self.anchor_models.vae.decode(SD_pred_latent.to(torch.float32) / 0.18215).sample
                    
                    SD_teacher_images = []
                    for i in SD_valid_indices:
                        teacher_latent = self.anchor_models.get_SD_teacher_latent(image_files[i][0])
                        teacher_image = self.anchor_models.vae.decode(teacher_latent.to(torch.float32) / 0.18215).sample
                        SD_teacher_images.append(teacher_image)
                    SD_teacher_images = torch.cat(SD_teacher_images, dim=0)
                    SD_teacher_images = SD_teacher_images.to(SD_pred_image.dtype)
                    
                    SD_vae_loss = self.anchor_loss.get_SD_image_align_loss(SD_pred_image, SD_teacher_images)
                else:
                    SD_vae_loss = 0.0
            SD_loss += SD_vae_loss
        else:
            SD_loss = 0.0
            
        if internvit_encoded_value is not None:
            # same as dino
            internvit_query = self.internvit_query_vectors.unsqueeze(0)
            bs = internvit_encoded_value.shape[0]
            internvit_mask = (input_ids == self.internvit_token_idx)
            if (~internvit_mask.bool()).all() == True:
                internvit_loss = 0.0
            else:
                last_hidden_state = outputs.hidden_states[-1]
                internvit_valid_indices = []
                internvit_hidden_features = []
                for i_ in range(bs):
                    if internvit_mask[i_].any():
                        internvit_valid_indices.append(i_)
                        internvit_features = last_hidden_state[i_, internvit_mask[i_]]
                        internvit_hidden_features.append(internvit_features)
                
                if len(internvit_hidden_features) > 0:
                    internvit_hidden_features = torch.stack(internvit_hidden_features, dim=0)
                    internvit_proj = nn.functional.normalize(self.internvit_projection(internvit_hidden_features)) # [B , 64 , 1024]
                    internvit_query_valid = internvit_query.expand(len(internvit_valid_indices), -1, -1).to(internvit_encoded_value.dtype)
                    internvit_attn_output, _ = self.internvit_cross_attention(
                        query=internvit_query_valid,        # Shape: [valid_batch_size, 1025, 1024]
                        key=internvit_proj,        # Shape: [valid_batch_size, 64, 1024]
                        value=internvit_proj       # Shape: [valid_batch_size, 64, 1024]
                    )
                    internvit_features_aligned = internvit_encoded_value[internvit_valid_indices].to(internvit_hidden_features.dtype)
                    # Compute MSE loss
                    internvit_loss = self.anchor_loss.get_internvit_loss(internvit_attn_output, internvit_features_aligned) # all of dino_features_aligned are same
                    internvit_loss = internvit_loss
                else:
                    internvit_loss = 0.0
            try:
                wandb.log({"internvit_loss": internvit_loss})
            except:
                pass
        else:
            internvit_loss = 0.0
            
        if pidinet_encoded_value is not None:
            pidinet_query = self.pidinet_query_vectors.unsqueeze(0)
            bs = pidinet_encoded_value.shape[0]
            pidinet_mask = (input_ids == self.pidinet_token_idx)

            if (~pidinet_mask.bool()).all() == True:
                pidinet_loss = 0.0
                
            else:
                last_hidden_state = outputs.hidden_states[-1]
                pidinet_valid_indices = []
                pidinet_hidden_features = []
                pidinet_mid_feats = []
                for i_ in range(bs):
                    if pidinet_mask[i_].any():
                        pidinet_valid_indices.append(i_)
                        pidinet_features = last_hidden_state[i_, pidinet_mask[i_]]
                        pidinet_hidden_features.append(pidinet_features)
                        pidinet_mid_feats.append(self.anchor_models.get_pidinet_mid_feats(image_files[i_][0]))
                
                if len(pidinet_hidden_features) > 0:
                    pidinet_hidden_features = torch.stack(pidinet_hidden_features, dim=0)
                    pidinet_proj = nn.functional.normalize(self.pidinet_projection(pidinet_hidden_features)) # [B , 64 , 1024]
                    pidinet_query_valid = pidinet_query.expand(len(pidinet_valid_indices), -1, -1).to(pidinet_encoded_value.dtype)
                    pidinet_attn_output, _ = self.pidinet_cross_attention(
                        query=pidinet_query_valid,        # Shape: [valid_batch_size, 60, 1024]
                        key=pidinet_proj,        # Shape: [valid_batch_size, 64, 1024]
                        value=pidinet_proj       # Shape: [valid_batch_size, 64, 1024]
                    )
                    pidinet_features_aligned = pidinet_encoded_value[pidinet_valid_indices].to(pidinet_hidden_features.dtype)
                    pidinet_mid_feats = torch.cat(pidinet_mid_feats, dim=0)
                    print(pidinet_mid_feats.shape)
                    pidinet_predicted_edge = self.anchor_models.get_edge_from_tokens(pidinet_attn_output, pidinet_mid_feats)
                    pidinet_features_aligned = torch.sigmoid(pidinet_features_aligned)
                    pidinet_loss = self.anchor_loss.get_pidinet_loss(pidinet_predicted_edge, pidinet_features_aligned)
                    pidinet_loss = pidinet_loss
                else:
                    pidinet_loss = 0.0
            try:
                wandb.log({"pidinet_loss": pidinet_loss})
            except:
                pass
        else:
            pidinet_loss = 0.0
            
        if siglip_encoded_value is not None:
            siglip_query = self.siglip_query_vectors.unsqueeze(0)
            bs = siglip_encoded_value.shape[0]
            siglip_mask = (input_ids == self.siglip_token_idx)
            if (~siglip_mask.bool()).all() == True:
                siglip_loss = 0.0
            else:
                last_hidden_state = outputs.hidden_states[-1]
                siglip_valid_indices = []
                siglip_hidden_features = []
                for i_ in range(bs):
                    if siglip_mask[i_].any():
                        siglip_valid_indices.append(i_)
                        siglip_features = last_hidden_state[i_, siglip_mask[i_]]
                        siglip_hidden_features.append(siglip_features)
                
                if len(siglip_hidden_features) > 0:
                    siglip_hidden_features = torch.stack(siglip_hidden_features, dim=0)
                    siglip_proj = nn.functional.normalize(self.siglip_projection(siglip_hidden_features)) # [B , 64 , 1024]
                    siglip_query_valid = siglip_query.expand(len(siglip_valid_indices), -1, -1).to(siglip_encoded_value.dtype)
                    siglip_attn_output, _ = self.siglip_cross_attention(
                        query=siglip_query_valid,        # Shape: [valid_batch_size, 1024, 1024]
                        key=siglip_proj,        # Shape: [valid_batch_size, 64, 1024]
                        value=siglip_proj       # Shape: [valid_batch_size, 64, 1024]
                    )
                    siglip_features_aligned = siglip_encoded_value[siglip_valid_indices].to(siglip_hidden_features.dtype)
                    siglip_loss = self.anchor_loss.get_siglip_loss(siglip_attn_output, siglip_features_aligned)
                    siglip_loss = siglip_loss
                else:
                    siglip_loss = 0.0
            try:
                wandb.log({"siglip_loss": siglip_loss})
            except:
                pass
        else:
            siglip_loss = 0.0
            
        if metaclip_encoded_value is not None:
            metaclip_query = self.metaclip_query_vectors.unsqueeze(0)
            bs = metaclip_encoded_value.shape[0]
            metaclip_mask = (input_ids == self.metaclip_token_idx)
            if (~metaclip_mask.bool()).all() == True:
                metaclip_loss = 0.0
            else:
                last_hidden_state = outputs.hidden_states[-1]
                metaclip_valid_indices = []
                metaclip_hidden_features = []
                for i_ in range(bs):
                    if metaclip_mask[i_].any():
                        metaclip_valid_indices.append(i_)
                        metaclip_features = last_hidden_state[i_, metaclip_mask[i_]]
                        metaclip_hidden_features.append(metaclip_features)
                
                if len(metaclip_hidden_features) > 0:
                    metaclip_hidden_features = torch.stack(metaclip_hidden_features, dim=0)
                    metaclip_proj = nn.functional.normalize(self.metaclip_projection(metaclip_hidden_features)) # [B , 64 , 1024]
                    metaclip_query_valid = metaclip_query.expand(len(metaclip_valid_indices), -1, -1).to(metaclip_encoded_value.dtype)
                    metaclip_attn_output, _ = self.metaclip_cross_attention(
                        query=metaclip_query_valid,        # Shape: [valid_batch_size, 1024, 1024]
                        key=metaclip_proj,        # Shape: [valid_batch_size, 64, 1024]
                        value=metaclip_proj       # Shape: [valid_batch_size, 64, 1024]
                    )
                    metaclip_features_aligned = metaclip_encoded_value[metaclip_valid_indices].to(metaclip_hidden_features.dtype)
                    metaclip_loss = self.anchor_loss.get_metaclip_loss(metaclip_attn_output, metaclip_features_aligned)
                    metaclip_loss = metaclip_loss
                else:
                    metaclip_loss = 0.0
            try:
                wandb.log({"metaclip_loss": metaclip_loss})
            except:
                pass
        else:
            metaclip_loss = 0.0
            
        if sam_encoded_value is not None or dino_encoded_value is not None or depth_encoded_value is not None or SD_encoded_value is not None or internvit_encoded_value is not None or pidinet_encoded_value is not None or siglip_encoded_value is not None or metaclip_encoded_value is not None:
            print(f'seg_loss: {seg_loss}, dino_loss: {dino_loss}, depth_loss: {depth_loss}, SD_loss: {SD_loss}, Internvit_loss: {internvit_loss}, pidinet_loss: {pidinet_loss}, siglip_loss: {siglip_loss}, metaclip_loss: {metaclip_loss}')
            
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if self.global_steps < self.align_anchor_task_only_stage:
            loss = seg_loss + dino_loss + depth_loss + SD_loss + internvit_loss + pidinet_loss + siglip_loss + metaclip_loss
        else:
            if labels is not None:
                # Upcast to float if we need to compute the loss to avoid potential precision issues
                logits = logits.float()
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                if self.global_steps <= self.align_vqa_only_stage:
                    loss = loss_fct(shift_logits, shift_labels) + seg_loss + dino_loss + depth_loss + SD_loss + internvit_loss + pidinet_loss + siglip_loss + metaclip_loss
                else:
                    loss = loss_fct(shift_logits, shift_labels)
            
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        
        anchor_outputs = (predict_masks_dict, predict_dino_dict, predict_depth_dict, predict_SD_dict, predict_internvit_dict, predict_pidinet_dict, predict_siglip_dict, predict_metaclip_dict)

        return Qwen2_5_VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
            anchor_outputs=anchor_outputs,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            use_cache=use_cache,
            **kwargs,
        )

        # Qwen2-5-VL position_ids are prepareed with rope_deltas in forward
        model_inputs["position_ids"] = None

        if cache_position[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None

        return model_inputs

    def _get_image_nums_and_video_nums(
        self,
        input_ids: Optional[torch.LongTensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the number of images and videos for each sample to calculate the separation length of the sample tensor.
        These parameters are not passed through the processor to avoid unpredictable impacts from interface modifications.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.

        Returns:
            image_nums (`torch.LongTensor` of shape `(batch_size, num_images_sample)`)
            video_nums (`torch.LongTensor` of shape `(batch_size, num_videos_sample)`)
        """
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id

        vision_start_mask = input_ids == vision_start_token_id
        vision_first_mask = torch.roll(vision_start_mask, shifts=1, dims=1)
        image_mask = input_ids == image_token_id
        video_mask = input_ids == video_token_id
        image_nums = torch.sum(vision_first_mask & image_mask, dim=1)
        video_nums = torch.sum(vision_first_mask & video_mask, dim=1)

        return image_nums, video_nums

    def _expand_inputs_for_generation(
        self,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: Optional[torch.LongTensor] = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        # Overwritten -- Support for expanding tensors without a batch size dimension
        # e.g., pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw, second_per_grid_t
        # pixel_values.shape[0] is sum(seqlen_images for samples)
        # image_grid_thw.shape[0] is sum(num_images for samples)

        if expand_size == 1:
            return input_ids, model_kwargs

        visual_keys = ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw", "second_per_grid_ts"]

        def _expand_dict_for_generation_visual(dict_to_expand):
            image_grid_thw = model_kwargs.get("image_grid_thw", None)
            video_grid_thw = model_kwargs.get("video_grid_thw", None)
            image_nums, video_nums = self._get_image_nums_and_video_nums(input_ids)

            def _repeat_interleave_samples(x, lengths, repeat_times):
                samples = torch.split(x, lengths)
                repeat_args = [repeat_times] + [1] * (x.dim() - 1)
                result = torch.cat([sample.repeat(*repeat_args) for sample in samples], dim=0)
                return result

            for key in dict_to_expand:
                if key == "pixel_values":
                    # split images into samples
                    samples = torch.split(image_grid_thw, list(image_nums))
                    # compute the sequence length of images for each sample
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "image_grid_thw":
                    # get the num of images for each sample
                    lengths = list(image_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "pixel_values_videos":
                    samples = torch.split(video_grid_thw, list(video_nums))
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "video_grid_thw":
                    lengths = list(video_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "second_per_grid_ts":
                    if not isinstance(dict_to_expand[key], list):
                        raise TypeError(
                            f"Expected value for key '{key}' to be a list, but got {type(dict_to_expand[key])} instead."
                        )
                    tensor = torch.tensor(dict_to_expand[key])
                    lengths = list(video_nums)
                    tensor = _repeat_interleave_samples(tensor, lengths=lengths, repeat_times=expand_size)
                    dict_to_expand[key] = tensor.tolist()
            return dict_to_expand

        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if (
                    key != "cache_position"
                    and dict_to_expand[key] is not None
                    and isinstance(dict_to_expand[key], torch.Tensor)
                    and key not in visual_keys
                ):
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
            return dict_to_expand

        # input_ids is required for expanding visual inputs
        # If input_ids is unavailable, visual inputs will not be used; therefore, there is no need to expand visual inputs.
        if input_ids is not None and input_ids.numel() != 0:
            model_kwargs = _expand_dict_for_generation_visual(model_kwargs)

        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        model_kwargs = _expand_dict_for_generation(model_kwargs)

        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])

        return input_ids, model_kwargs
