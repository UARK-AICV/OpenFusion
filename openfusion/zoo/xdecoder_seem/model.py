# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------

import os
import logging
import torch
import torch.nn as nn
from detectron2.data import MetadataCatalog
from openfusion.zoo.xdecoder_seem.utils.constants import COCO_PANOPTIC_CLASSES
from openfusion.zoo.xdecoder_seem.utils.model_loading import align_and_update_state_dicts

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class BaseModel(nn.Module):
    def __init__(self, opt, module: nn.Module):
        super(BaseModel, self).__init__()
        self.opt = opt
        self.model = module

    @torch.inference_mode()
    def init_vocabulary(self, vocab=COCO_PANOPTIC_CLASSES + ["background"]):
        self.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
            vocab, is_eval=True
        )
        metadata = MetadataCatalog.get('coco_2017_train_panoptic')
        self.model.metadata = metadata

    @torch.inference_mode()
    def forward(self, *inputs, **kwargs):
        outputs = self.model(*inputs, **kwargs)
        res_list = [
            outputs[i]["seg"] for i in range(len(outputs))
        ]
        return res_list

    @torch.inference_mode()
    def encode_text(self, texts):
        return self.model.sem_seg_head.predictor.lang_encoder.get_text_token_embeddings(
            texts
        )["class_emb"]

    def save_pretrained(self, save_dir):
        save_path = os.path.join(save_dir, 'model_state_dict.pt')
        torch.save(self.model.state_dict(), save_path)

    def from_pretrained(self, load_path):
        state_dict = torch.load(load_path, map_location=self.opt['device'])
        state_dict = align_and_update_state_dicts(self.model.state_dict(), state_dict)
        self.model.load_state_dict(state_dict, strict=False)
        return self
