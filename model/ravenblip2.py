# from src.models.base import BaseModel
# from lavis.models.blip2_models.blip2_t5_instruct import Blip2T5Instruct
from model.ravenblip import RavenBlip2T5Instruct
from lavis.models import load_preprocess
import torch
from omegaconf import OmegaConf
from PIL import Image

import torch.nn as nn
from abc import abstractmethod

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.prompt = None

    def forward(self, *args, **kwargs):
        self.model.forward(*args, **kwargs)

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    @abstractmethod
    def parse_images(self, images):
        raise NotImplementedError

    def parse_text_input(self, questions, prompt=None):
        if not isinstance(questions, list):
            questions = [questions]
        if not prompt:
            return questions
        return [prompt.format(question) for question in questions]


class RavenBlip2(BaseModel):
    def __init__(self, config_path):
        super().__init__()
        model_default_config = OmegaConf.load(config_path)
        self.model = RavenBlip2T5Instruct(model_default_config)

        self.vis_processors, self.text_processors = load_preprocess(
            model_default_config.preprocess
        )
        self.model.load_from_pretrained(model_default_config.custom.model_path)

    # def train(self):model_path
    #     # self.model.Qformer.train()
    #     # self.model.t5_proj.train()
    #     self.model.affordance_head.train()

    # def eval(self):
    #     self.model.eval()

    def get_params(self, weight_decay, lr_scale=1):
        weight_decay_params, non_weight_decay_params = [], []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim < 2 or "bias" in name or "ln" in name or "bn" in name:
                non_weight_decay_params.append(param)
            else:
                weight_decay_params.append(param)

        optim_params = [
            {
                "params": weight_decay_params,
                "weight_decay": weight_decay,
                "lr_scale": lr_scale,
            },
            {
                "params": non_weight_decay_params,
                "weight_decay": 0,
                "lr_scale": lr_scale,
            },
        ]

        return optim_params

    def parse_images(self, images):
        if not isinstance(images, list):
            images = [images]
        # try:
        images = torch.stack(
            [self.vis_processors["eval"](x) for x in images], dim=0
        ).to(self.device)
        # else:
        # images = torch.stack(
        #     [x for x in images], dim=0
        # ).to(self.device)
        return images

    def parse_questions(self, questions, mode="train"):
        return [self.text_processors[mode](question) for question in questions]

    # TODO: move this procedure to dataset
    def forward(self, batch):
        images = self.parse_images(batch["init_image"])
        # final_images = self.parse_images(batch["final_image"])
        instruction = self.parse_questions(batch["instruction"])
        samples = {
            "init_image": images,
            "text_input": instruction,
        }

        if "n_answers" in batch:
            samples["n_answers"] = batch["n_answers"]
        return self.model(samples=samples)
    
    @torch.no_grad()
    def generate(self, batch):
        images = self.parse_images(batch["init_image"])
        instruction = self.parse_questions(batch["instruction"])
        samples = {
            "init_image": images,
            "text_input": instruction,
        }

        if "n_answers" in batch:
            samples["n_answers"] = batch["n_answers"]
        return self.model(samples=samples)

if __name__ == "__main__":
    
    config_path = './config.yaml'
    # model_path = './instruct_blip_flanxl_trimmed.pth'

    image1 = Image.open('./0.png').convert("RGB")
    image2 = Image.open('./0.png').convert("RGB")
    # images = [image1, image2]
    images = [image1]
    # questions = ["What is the color of the object?", "What is the shape of the object?"]
    questions = ["What is the color of the object?"]
    batch = {
        "init_image": images,
        "instruction": questions,
    }
    blip2 = RavenBlip2(config_path=config_path)
    blip2 = blip2.cuda()
    blip2.eval()
    a = blip2(batch)
    print(a.shape)
