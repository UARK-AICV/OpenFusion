# --------------------------------------------------------
# SEEM -- Segment Everything Everywhere All At Once
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------

import random
from typing import Tuple
from torchvision import transforms
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import cv2  # type: ignore
from .registry import register_model
from ..utils import configurable
from ..utils import get_iou
from ..backbone import build_backbone, Backbone
from ..body import build_xdecoder_head
from ..modules import sem_seg_postprocess, bbox_postprocess
from ..language import build_language_encoder
from ..language.loss import vl_similarity

from nltk.stem.lancaster import LancasterStemmer
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.data import MetadataCatalog
try:
    import torchshow as ts
    import matplotlib.pyplot as plt
    import seaborn as sns
except:
    pass

st = LancasterStemmer()

kernel = transforms.GaussianBlur(5)

class SEEM_Model(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        losses: dict,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        task_switch: dict,
        phrase_prob: float,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
        train_dataset_name: str,
        interactive_mode: str,
        interactive_iter: str,
        dilation_kernel: torch.Tensor,
    ):
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.losses = losses
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # caption argument
        self.task_switch = task_switch
        self.phrase_prob = phrase_prob

        self.test_topk_per_image = test_topk_per_image
        self.train_class_names = None
        self.interactive_mode = interactive_mode
        self.interactive_iter = interactive_iter

        self.register_buffer("dilation_kernel", dilation_kernel)

    @classmethod
    def from_config(cls, cfg):
        enc_cfg = cfg['MODEL']['ENCODER']
        dec_cfg = cfg['MODEL']['DECODER']

        openimage_switch = {'grounding': dec_cfg['OPENIMAGE']['GROUNDING'].get('ENABLED', False),
                            'mask': dec_cfg['OPENIMAGE'].get('ENABLED', False)}

        task_switch = {'bbox': dec_cfg.get('DETECTION', False),
                       'mask': dec_cfg.get('MASK', True),
                       'spatial': dec_cfg['SPATIAL'].get('ENABLED', False),
                       'grounding': dec_cfg['GROUNDING'].get('ENABLED', False),
                       'openimage': openimage_switch,
                       'visual': dec_cfg['VISUAL'].get('ENABLED', False),
                       'audio': dec_cfg['AUDIO'].get('ENABLED', False)}

        # build model
        extra = {'task_switch': task_switch}
        backbone = build_backbone(cfg)
        lang_encoder = build_language_encoder(cfg)
        sem_seg_head = build_xdecoder_head(cfg, backbone.output_shape(), lang_encoder, extra=extra)

        # Training Settings.
        loss_weights = {}
        matcher = None
        losses = {}
        weight_dict = {}
        grd_weight = {}
        top_x_layers = {}
        criterion = None
        train_dataset_name = None
        phrase_prob = None
        # Loss parameters:
        deep_supervision = None
        no_object_weight = None

        interactive_mode = 'best'
        interactive_iter = 20
        dilation = 3
        dilation_kernel = torch.ones((1, 1, dilation, dilation), device=torch.cuda.current_device())

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "losses": losses,
            "num_queries": dec_cfg['NUM_OBJECT_QUERIES'],
            "object_mask_threshold": dec_cfg['TEST']['OBJECT_MASK_THRESHOLD'],
            "overlap_threshold": dec_cfg['TEST']['OVERLAP_THRESHOLD'],
            "metadata": None,
            "size_divisibility": dec_cfg['SIZE_DIVISIBILITY'],
            "sem_seg_postprocess_before_inference": (
                dec_cfg['TEST']['SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE']
                or dec_cfg['TEST']['PANOPTIC_ON']
                or dec_cfg['TEST']['INSTANCE_ON']
            ),
            "pixel_mean": cfg['INPUT']['PIXEL_MEAN'],
            "pixel_std": cfg['INPUT']['PIXEL_STD'],
            "task_switch": task_switch,
            "phrase_prob": phrase_prob,
            # inference
            "semantic_on": dec_cfg['TEST']['SEMANTIC_ON'],
            "instance_on": dec_cfg['TEST']['INSTANCE_ON'],
            "panoptic_on": dec_cfg['TEST']['PANOPTIC_ON'],
            "test_topk_per_image": cfg['MODEL']['DECODER']['TEST']['DETECTIONS_PER_IMAGE'],
            "train_dataset_name": train_dataset_name,
            "interactive_mode": interactive_mode,
            "interactive_iter": interactive_iter,
            "dilation_kernel": dilation_kernel,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    @torch.inference_mode()
    def forward(self, images, mode='default'):
        images = (images - self.pixel_mean) / self.pixel_std
        targets = targets_grounding = queries_grounding = None
        features = self.backbone(images)
        outputs = self.sem_seg_head(features, target_queries=queries_grounding)

        mask_pred_captions = outputs['pred_captions']
        mask_pred_results = outputs["pred_masks"]

        # upsample masks
        # mask_pred_results = F.interpolate(
        #     mask_pred_results,
        #     size=(images.shape[-2], images.shape[-1]),
        #     mode="bilinear",
        #     align_corners=False,
        # )
        del outputs
        # ts.show(images[0])

        processed_results = []
        for mask_pred_result, mask_pred_caption in zip(
            mask_pred_results, mask_pred_captions
        ):
            processed_results.append({})
            mask_cls_result = self.sem_seg_head.predictor.lang_encoder.compute_similarity(mask_pred_caption)[0]

            if mode == "default":
                panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result, mask_pred_caption)
                processed_results[-1]["seg"] = panoptic_r

            elif mode == "semseg":
                semantic_r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                processed_results[-1]["seg"] = semantic_r

            elif mode == "emb":
                res = torch.einsum(
                    "qd,qhw->dhw",
                    mask_pred_caption / (mask_pred_caption.norm(dim=-1, keepdim=True) + 1e-7),
                    mask_pred_result.sigmoid()
                ).permute(1,2,0) # h,w,d
                processed_results[-1]["seg"] = {"emb": res}

                """"""
                # TODO: remove this
                from openfusion.zoo.xdecoder_seem.utils.constants import COCO_PANOPTIC_CLASSES
                t_emb = self.sem_seg_head.predictor.lang_encoder.get_text_token_embeddings(
                    COCO_PANOPTIC_CLASSES
                )["class_emb"]
                t_emb = t_emb / (t_emb.norm(dim=-1, keepdim=True) + 1e-7)
                processed_results[-1]["seg"]["t_emb"] = t_emb
                # out = torch.einsum("cd,hwd->chw", t_emb, res)
                # ts.show(out.argmax(0))
                """"""
        return processed_results

    @staticmethod
    def remove_small_regions(
        mask: np.ndarray, area_thresh: float, mode: str
    ) -> Tuple[np.ndarray, bool]:
        """
        Removes small disconnected regions and holes in a mask. Returns the
        mask and an indicator of if the mask has been modified.
        """
        assert mode in ["holes", "islands"]
        correct_holes = mode == "holes"
        working_mask = (correct_holes ^ mask).astype(np.uint8)
        n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(working_mask, 8)
        sizes = stats[:, -1][1:]  # Row 0 is background label
        small_regions = [i + 1 for i, s in enumerate(sizes) if s < area_thresh]
        if len(small_regions) == 0:
            return mask, False
        fill_labels = [0] + small_regions
        if not correct_holes:
            fill_labels = [i for i in range(n_labels) if i not in fill_labels]
            # If every region is below threshold, keep largest
            if len(fill_labels) == 0:
                fill_labels = [int(np.argmax(sizes)) + 1]
        mask = np.isin(regions, fill_labels)
        return mask, True

    def postprocess_small_region(self, mask, min_area:int=0):
        """
        Removes small disconnected regions and holes in masks
        Requires open-cv as a dependency.
        """
        if min_area == 0:
            return mask
        # Filter small disconnected regions and holes
        mask = mask.cpu().numpy()
        mask, _ = self.remove_small_regions(mask, min_area, mode="holes")
        mask, _ = self.remove_small_regions(mask, min_area, mode="islands")
        return torch.as_tensor(mask).to(self.device)

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return {
            'conf_idx': semseg.argmax(0),
            'conf_score': semseg.max(0)[0],
            'caption': self.sem_seg_head.predictor.lang_encoder.default_text_embeddings
        }

    def panoptic_inference(self, mask_cls, mask_pred, mask_pred_caption):
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()

        keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold) # 0.4
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        caption_indices = torch.nonzero(keep).flatten()
        cur_captions = mask_pred_caption[caption_indices] # Shape: len(masks), 512

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), device=cur_masks.device).long()
        panoptic_conf = torch.zeros((h, w), device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # NOTE: didn't detect any mask :(
            return {
                'conf_idx': panoptic_seg,
                'conf_score': panoptic_conf,
                # 'segments': segments_info,
                'caption': torch.zeros(0, 512)
            }
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask_ = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)
                mask = self.postprocess_small_region(mask_, min_area=50) # min_area=100

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        # print(f"low overlap {mask_area / original_area} below {self.overlap_threshold}")
                        continue

                    isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                    # NOTE: merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id
                    panoptic_conf[mask] = cur_masks[k][mask]

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                            "caption": cur_captions[k],
                            "confidence_score": cur_masks[k] * mask
                        }
                    )

            # NOTE: as panoptic_seg id starts from 1, shift by 1 to use as index for caption
            caption = torch.stack([s['caption'] for s in segments_info]) if len(segments_info) != 0 else torch.zeros(0, 512)
            return {
                'conf_idx': panoptic_seg,
                'conf_score': panoptic_conf,
                # 'category_id': torch.tensor([s['category_id'] for s in segments_info]) if len(segments_info) != 0 else torch.zeros(0),
                'caption': caption
            }


@register_model
def get_segmentation_model(cfg, **kwargs):
    return SEEM_Model(cfg)
