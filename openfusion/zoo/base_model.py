import os
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from PIL import Image
from typing import Any


BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODELS = {
    "seem": {
        "input_size": 360,
        "config": f"{BASE_PATH}/xdecoder_seem/configs/seem/seem_focall_lang.yaml",
        "checkpoint": f"{BASE_PATH}/xdecoder_seem/checkpoints/seem_focall_v1.pt",
    },
}


class VLFM(nn.Module):
    def __init__(self, name, **kwargs) -> None:
        super().__init__()
        self.name = name
        self.meta = MODELS[name]
        self.transform = transforms.Resize(
            kwargs.get("input_size", self.meta["input_size"]),
            interpolation=Image.BICUBIC
        )

    def encode_prompt(self, prompt, task="default"):
        if task == "default":
            return self.encode_text(prompt)

    def encode_text(self, text):
        pass

    def preprocess_image(self, rgb):
        pass

    def encode_image(self, image):
        pass


class ImageAlignedModel(VLFM):
    def __init__(self, **kwargs) -> None:
        super().__init__()

    def encode_text(self, text):
        pass

    def preprocess_image(self, rgb):
        pass

    def encode_image(self, image):
        pass


class RegionAlignedModel(VLFM):
    def __init__(self, name, **kwargs) -> None:
        super().__init__(name, **kwargs)
        if name == "seem":
            from openfusion.zoo.xdecoder_seem.model import BaseModel
            from openfusion.zoo.xdecoder_seem.xdecoder import build_model
            from openfusion.zoo.xdecoder_seem.utils.distributed import init_distributed
            from openfusion.zoo.xdecoder_seem.utils.arguments import load_opt_from_config_files
            opt = load_opt_from_config_files(self.meta["config"])
            opt = init_distributed(opt)
            self.model = BaseModel(opt, build_model(opt)).from_pretrained(
                self.meta["checkpoint"]
            ).eval().cuda()
            self.model.init_vocabulary()
        else:
            raise NotImplementedError
        print("[*] model loaded")

    @staticmethod
    def model_names():
        return ["seem"]

    @property
    def dim(self):
        return 512

    @torch.inference_mode()
    def encode_text(self, texts):
        return self.model.encode_text(texts)

    def preprocess_image(self, rgb):
        images = [np.asarray(self.transform(Image.fromarray(i))) for i in rgb]
        # NOTE: normalize image inside model
        images = torch.tensor(
            np.asarray(images, dtype=np.float32)
        ).float().permute(0, 3, 1, 2).cuda()
        return images

    @torch.inference_mode()
    def encode_image(self, rgb, mode="default"):
        rgb_images = self.preprocess_image(rgb)
        assert rgb_images.shape[1] == 3
        return self.model(rgb_images, mode)


class PixelAlignedModel(VLFM):
    def __init__(self, **kwargs) -> None:
        super().__init__()

    def encode_text(self, text):
        pass

    def preprocess_image(self, rgb):
        pass

    def encode_image(self, image):
        pass


def build_vl_model(name, **kwargs):
    if name in MODELS:
        if name in RegionAlignedModel.model_names():
            return RegionAlignedModel(name, **kwargs)
        else:
            raise ValueError(f"[*] model {name} not implemented")
    else:
        raise NotImplementedError(f"[*] model {name} not implemented")