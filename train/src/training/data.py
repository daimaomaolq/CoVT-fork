import ast
import copy
import os
from dataclasses import dataclass, field
from typing import Dict
import torch
import transformers
import ujson as json
from torch.utils.data import Dataset
from qwen_vl_utils import process_vision_info
from PIL import Image
import re
import numpy as np
import random

from .params import DataArguments
from .constants import *


def truncate_sequence(input_ids, labels, max_length, eos_token_id):
    if input_ids.size(0) > max_length:
        input_ids = input_ids[:max_length-1]
        labels = labels[:max_length-1]

    if eos_token_id is not None:
        input_ids = torch.cat([input_ids, torch.tensor([eos_token_id])])
        labels = torch.cat([labels, torch.tensor([eos_token_id])])

    return input_ids, labels

def pad_sequence(sequences, padding_side='right', padding_value=0):
    """
    Pad a list of sequences to the same length.
    sequences: list of tensors in [seq_len, *] shape
    """
    assert padding_side in ['right', 'left']
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == 'right':
            output.data[i, :length] = seq
        else:
            output.data[i, -length:] = seq
    return output

def get_image_info(image_path, min_pixel, max_pixel, width, height):
    # Using this because of process_vision_info function
    # Need to fix this in the future


    content = {
        "type": "image", 
        "image": image_path,
        "min_pixels": min_pixel,
        "max_pixels": max_pixel
    }

    if width is not None and height is not None:
        content["resized_width"] = width
        content["resized_height"] = height

    messages = [
        {"role": "user", 
         "content": [content]
        }
    ]

    image_input, _ = process_vision_info(messages)

    return image_input[0]

def get_video_info(video_path, min_pixels, max_pixels, fps):
    # Using this because of process_vision_info function
    # Need to fix this in the future

    messages = [
        {"role": "user", 
         "content": [
             {
                "type": "video", 
                "video": video_path,
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
                "fps": fps
            }
            ]
        }
    ]

    _, video_input, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

    return video_input[0], video_kwargs

def build_anchor_pads(anchor_nums, anchor_tokens):
    anchor_pads = []
    for anchor_num, anchor_token in zip(anchor_nums, anchor_tokens):
        anchor_pad = ANCHOR_START_TOKEN + anchor_token * anchor_num + ANCHOR_END_TOKEN
        anchor_pads.append(anchor_pad)
    return "".join(anchor_pads)

def add_anchor_pad(user_input, anchor_nums, anchor_tokens):
    # Legacy CoVT behavior: add anchor pads immediately after image tokens.
    anchor_pads = build_anchor_pads(anchor_nums, anchor_tokens)
    if VISION_END_TOKEN in user_input:
        user_input = user_input.replace(VISION_END_TOKEN, VISION_END_TOKEN + anchor_pads)
    return user_input

def add_anchor_pad_after_query(user_input, anchor_nums, anchor_tokens):
    # Causal decoders can only condition anchor-token states on the query when
    # the anchor tokens appear after the query and before the assistant turn.
    anchor_pads = build_anchor_pads(anchor_nums, anchor_tokens)
    if not anchor_pads:
        return user_input

    assistant_start = f"{DEFAULT_IM_START_TOKEN}assistant\n"
    assistant_pos = user_input.rfind(assistant_start)
    search_end = assistant_pos if assistant_pos >= 0 else len(user_input)
    user_end_pos = user_input.rfind(DEFAULT_IM_END_TOKEN, 0, search_end)
    prefix_source = user_input[:user_end_pos] if user_end_pos >= 0 else user_input
    anchor_prefix = "" if prefix_source.endswith("\n") else "\n"
    anchor_text = anchor_prefix + "Visual anchors: " + anchor_pads + "\n"

    if user_end_pos >= 0:
        return user_input[:user_end_pos] + anchor_text + user_input[user_end_pos:]
    if VISION_END_TOKEN in user_input:
        return user_input.replace(VISION_END_TOKEN, VISION_END_TOKEN + anchor_text, 1)
    return user_input + anchor_text

def add_anchor_pad_with_mode(user_input, anchor_nums, anchor_tokens, mode):
    mode = (mode or "legacy").lower()
    if mode in ("none", "off", "false"):
        return user_input
    if mode in ("legacy", "after_vision", "vision"):
        return add_anchor_pad(user_input, anchor_nums, anchor_tokens)
    if mode in ("query_tail", "after_query", "before_user_end"):
        return add_anchor_pad_after_query(user_input, anchor_nums, anchor_tokens)
    raise ValueError(f"Unknown anchor_prompt_mode: {mode}")


def add_cot_anchor_pad_in_user_input(user_input, anchor_nums, anchor_tokens):
    if len(anchor_nums) == 0:
        return user_input
    
    anchor_pads = []
    for anchor_num, anchor_token in zip(anchor_nums, anchor_tokens, anchor_names):
        anchor_pad = ANCHOR_START_TOKEN + anchor_token * anchor_num + ANCHOR_END_TOKEN
        anchor_pads.append(anchor_pad)
    CoT_pad = ""
    if len(anchor_pads) == 1:
        CoT_pad = f"The {anchor_names[0]} of the image is {anchor_pads[0]}. "
    else:
        for i, (anchor_name, anchor_pad) in enumerate(zip(anchor_names, anchor_pads)):
            if i == 0:
                CoT_pad += f"The {anchor_name} of the image is {anchor_pad}, "
            elif i == len(anchor_names) - 1:
                CoT_pad += f"and the {anchor_name} of the image is {anchor_pad}. "
            else:
                CoT_pad += f"the {anchor_name} of the image is {anchor_pad}, "
    user_input = CoT_pad + user_input
    return user_input

def get_cot_data_in_response(response, anchor_nums, anchor_tokens, anchor_names):
    if len(anchor_nums) == 0:
        return response
    
    anchor_pads = []
    for anchor_num, anchor_token in zip(anchor_nums, anchor_tokens):
        anchor_pad = ANCHOR_START_TOKEN + anchor_token * anchor_num + ANCHOR_END_TOKEN
        anchor_pads.append(anchor_pad)
    CoT_start = "Because "
    if len(anchor_names) == 1:
        CoT_start += f"the {anchor_names[0]} of the image is {anchor_pads[0]}. "
    else:
        for anchor_name, anchor_pad in zip(anchor_names, anchor_pads):
            CoT_start += f"the {anchor_name} of the image is {anchor_pad}"
            if anchor_name == anchor_names[-2]:
                CoT_start += ", and "
            elif anchor_name == anchor_names[-1]:
                CoT_start += ". "
            else:
                CoT_start += ", "
    response = CoT_start + response
    return response


COT_TEMPLATES = [
    {
        "name": "basic_causal",
        "single": "Because the {anchor_name} of the image is {anchor_pad}. ",
        "multiple": "Because the {anchor_name} of the image is {anchor_pad}{connector}",
        "connectors": {
            "middle": ", ",
            "second_last": ", and ",
            "last": ". "
        }
    },
    
    {
        "name": "observational",
        "single": "I can observe that the {anchor_name} of the image is {anchor_pad}. ",
        "multiple": "I can observe that the {anchor_name} of the image is {anchor_pad}{connector}",
        "connectors": {
            "middle": ", ",
            "second_last": ", and ",
            "last": ". "
        }
    },
    
    {
        "name": "analytical",
        "single": "After analyzing the image, the {anchor_name} is {anchor_pad}. ",
        "multiple": "After analyzing the image, the {anchor_name} is {anchor_pad}{connector}",
        "connectors": {
            "middle": ", ",
            "second_last": ", and ",
            "last": ". "
        }
    },
    
    {
        "name": "descriptive",
        "single": "The image shows that the {anchor_name} is {anchor_pad}. ",
        "multiple": "The image shows that the {anchor_name} is {anchor_pad}{connector}",
        "connectors": {
            "middle": ", ",
            "second_last": ", and ",
            "last": ". "
        }
    },
    
    {
        "name": "conditional",
        "single": "Given that the {anchor_name} of the image is {anchor_pad}. ",
        "multiple": "Given that the {anchor_name} of the image is {anchor_pad}{connector}",
        "connectors": {
            "middle": ", ",
            "second_last": ", and ",
            "last": ". "
        }
    },
    
    {
        "name": "evidence_based",
        "single": "Based on the visual evidence, the {anchor_name} of the image is {anchor_pad}. ",
        "multiple": "Based on the visual evidence, the {anchor_name} of the image is {anchor_pad}{connector}",
        "connectors": {
            "middle": ", ",
            "second_last": ", and ",
            "last": ". "
        }
    }
]

def get_random_cot_template():
    return random.choice(COT_TEMPLATES)

def apply_cot_template(template, anchor_names, anchor_pads):
    if len(anchor_names) == 1:
        return template["single"].format(
            anchor_name=anchor_names[0], 
            anchor_pad=anchor_pads[0]
        )
    else:
        result = ""
        for i, (anchor_name, anchor_pad) in enumerate(zip(anchor_names, anchor_pads)):
            if i == len(anchor_names) - 1: 
                connector = template["connectors"]["last"]
            elif i == len(anchor_names) - 2: 
                connector = template["connectors"]["second_last"]
            else: 
                connector = template["connectors"]["middle"]
            
            result += template["multiple"].format(
                anchor_name=anchor_name,
                anchor_pad=anchor_pad,
                connector=connector
            )
        return result

def get_templates_comt_data_in_response(response, anchor_nums, anchor_tokens, anchor_names):
    if len(anchor_nums) == 0:
        return response
    
    anchor_pads = []
    for anchor_num, anchor_token in zip(anchor_nums, anchor_tokens):
        anchor_pad = ANCHOR_START_TOKEN + anchor_token * anchor_num + ANCHOR_END_TOKEN
        anchor_pads.append(anchor_pad)
    
    template = get_random_cot_template()
    
    cot_text = apply_cot_template(template, anchor_names, anchor_pads)
    
    response = "<think>" + cot_text + "</think>" + "<answer>" + response + "</answer>"
    return response
        
def get_comt_data_in_response(response, anchor_nums, anchor_tokens, anchor_names):
    if len(anchor_nums) == 0:
        return response
    
    anchor_pads = []
    for anchor_num, anchor_token in zip(anchor_nums, anchor_tokens):
        anchor_pad = ANCHOR_START_TOKEN + anchor_token * anchor_num + ANCHOR_END_TOKEN
        anchor_pads.append(anchor_pad)
    CoT_start = "<think> Because "
    if len(anchor_names) == 1:
        CoT_start += f"the {anchor_names[0]} of the image is {anchor_pads[0]}. "
    else:
        for anchor_name, anchor_pad in zip(anchor_names, anchor_pads):
            CoT_start += f"the {anchor_name} of the image is {anchor_pad}"
            if anchor_name == anchor_names[-2]:
                CoT_start += ", and "
            elif anchor_name == anchor_names[-1]:
                CoT_start += ". "
            else:
                CoT_start += ", "
    response = CoT_start + " </think>\n" + "<answer> " + response + " </answer>"
    return response
    
def get_feature_data(user_input, gpt_response, anchor_nums, anchor_tokens, anchor_names):
    anchor_pads = []
    for anchor_num, anchor_token, anchor_name in zip(anchor_nums, anchor_tokens, anchor_names):
        anchor_pad = ANCHOR_START_TOKEN + anchor_token * anchor_num + ANCHOR_END_TOKEN
        anchor_pads.append(anchor_pad)
    anchor_name = ", ".join(anchor_names)
    anchor_pads = "".join(anchor_pads)
    user_input = f"{DEFAULT_IM_START_TOKEN}{user_input['role']}\n{VISION_START_TOKEN + DEFAULT_IMAGE_TOKEN + VISION_END_TOKEN}What is the {anchor_name} of the image?\n{DEFAULT_IM_END_TOKEN}\n{DEFAULT_IM_START_TOKEN}{gpt_response['role']}\n"
    gpt_response = f"{anchor_pads}\n{DEFAULT_IM_END_TOKEN}\n"
    return user_input, gpt_response

def replace_pad_with_anchor_tokens(gpt_response):
    token_dict = {
        "<segmentation>": SAM_PAD_TOKEN * 8,
        "<depth>": DEPTH_PAD_TOKEN * 4,
        "<dino>": DINO_PAD_TOKEN * 4,
        "<pidinet>": PIDINET_PAD_TOKEN * 4,
        "<siglip>": SIGLIP_PAD_TOKEN * 4,
        "<metaclip>": METACLIP_PAD_TOKEN * 4,
    }
    for token, anchor_token in token_dict.items():
        gpt_response = gpt_response.replace(token, anchor_token)
    return gpt_response
    
CANONICAL_ANCHOR_ORDER = ["sam", "dino", "depth", "SD", "InternViT", "pidinet", "siglip", "metaclip"]
DEFAULT_ANCHOR_TOKEN_COUNTS = {
    "sam": 8,
    "dino": 4,
    "depth": 4,
    "SD": 4,
    "InternViT": 4,
    "pidinet": 4,
    "siglip": 4,
    "metaclip": 4,
}


def parse_anchor_token_counts(anchor_model_id, raw_counts=None):
    if raw_counts is None or raw_counts == "":
        return [DEFAULT_ANCHOR_TOKEN_COUNTS[anchor_model] for anchor_model in anchor_model_id]
    if isinstance(raw_counts, str):
        raw_counts = ast.literal_eval(raw_counts)
    counts = [int(value) for value in raw_counts]
    if len(counts) == len(CANONICAL_ANCHOR_ORDER):
        count_by_id = dict(zip(CANONICAL_ANCHOR_ORDER, counts))
        return [count_by_id[anchor_model] for anchor_model in anchor_model_id]
    if len(counts) == len(anchor_model_id):
        return counts
    raise ValueError(
        "anchor_token_counts must contain either 8 canonical values "
        "or one value per selected anchor model"
    )


def get_token_num(anchor_model_id, raw_counts=None):
    return parse_anchor_token_counts(anchor_model_id, raw_counts)
def get_anchor_token(anchor_model_id):
    anchor_tokens = []
    for anchor_model in anchor_model_id:
        if anchor_model == "sam":
            anchor_tokens.append(SAM_PAD_TOKEN)
        elif anchor_model == "dino":
            anchor_tokens.append(DINO_PAD_TOKEN)
        elif anchor_model == "depth":
            anchor_tokens.append(DEPTH_PAD_TOKEN)
        elif anchor_model == "SD":
            anchor_tokens.append(SD_PAD_TOKEN)
        elif anchor_model == "InternViT":
            anchor_tokens.append(INTERN_PAD_TOKEN)
        elif anchor_model == "pidinet":
            anchor_tokens.append(PIDINET_PAD_TOKEN)
        elif anchor_model == "siglip":
            anchor_tokens.append(SIGLIP_PAD_TOKEN)
        elif anchor_model == "metaclip":
            anchor_tokens.append(METACLIP_PAD_TOKEN)
    return anchor_tokens

def get_anchor_task_name(anchor_model_id):
    anchor_task_names = []
    for anchor_model in anchor_model_id:
        if anchor_model == "sam":
            anchor_task_names.append("segmentation")
        elif anchor_model == "dino":
            anchor_task_names.append("perception feature")
        elif anchor_model == "depth":
            anchor_task_names.append("depth map")
        elif anchor_model == "SD":
            anchor_task_names.append("style")
        elif anchor_model == "InternViT":
            anchor_task_names.append("caption")
        elif anchor_model == "pidinet":
            anchor_task_names.append("edge map")
        elif anchor_model == "siglip":
            anchor_task_names.append("clip feature")
        elif anchor_model == "metaclip":
            anchor_task_names.append("metaclip feature")
    return anchor_task_names
    

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str | list,
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        model_id,
        padding=True,
        shuffle=True,
        random_seed=42,
        anchor_model_id=None,
    ):
        super(SupervisedDataset, self).__init__()
        if isinstance(data_path, str):
            list_data_dict = json.load(open(data_path, "r"))
        else:
            list_data_dict = data_path

        self.model_id = model_id
        self.processor = processor
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.padding = padding
        self.image_min_pixel = data_args.image_min_pixels
        self.image_max_pixel = data_args.image_max_pixels
        self.image_resized_w = data_args.image_resized_width
        self.image_resized_h = data_args.image_resized_height
        self.video_min_pixel = data_args.video_min_pixels
        self.video_max_pixel = data_args.video_max_pixels
        self.fps = data_args.fps
        self.anchor_model_id = anchor_model_id
        self.anchor_token_nums = get_token_num(anchor_model_id, getattr(data_args, "anchor_token_counts", None))
        self.anchor_tokens = get_anchor_token(anchor_model_id)
        self.anchor_task_names = get_anchor_task_name(anchor_model_id)
        
        self.cur_step = 0
        self.stage_0_step = data_args.stage_0_step
        self.stage_1_step = data_args.stage_1_step
        self.stage_2_step = data_args.stage_2_step
        self.anchor_prompt_mode = (data_args.anchor_prompt_mode or "legacy").lower()
        self.anchor_response_mode = (data_args.anchor_response_mode or "legacy").lower()
        
        # for shuffle
        self.rng = np.random.default_rng(seed=random_seed)
        
        if shuffle:
            self.rng.shuffle(self.list_data_dict)
    
    def set_cur_step(self, step: int):
        self.cur_step = step
        print(f"[Dataset] cur_step has been set to {step}")
        
    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        
        # import ipdb; ipdb.set_trace()
        
        sources = self.list_data_dict[i]
        
        is_video = False

        processor = self.processor
        if "image" in sources:
            videos = None
            grid_key = "image_grid_thw"
            pixel_key = "pixel_values"
            
            image_files = sources["image"]
            image_folder = self.data_args.image_folder

            def resolve_image_file(image_file):
                if isinstance(image_file, str) and image_folder and not image_file.startswith("http") and not os.path.isabs(image_file):
                    candidate = os.path.join(image_folder, image_file)
                    if os.path.exists(candidate):
                        return candidate
                return image_file

            if isinstance(image_files, str):
                image_files = Image.open(resolve_image_file(image_files)).convert("RGB")
                image_files = [image_files]
            else:
                image_files = [Image.open(resolve_image_file(image_file)).convert("RGB") for image_file in image_files]

            images = []
            
            for image_file in image_files:
                
                # if not os.path.exists(image_file):
                #     if not image_file.startswith("http"):
                #         image_file = os.path.join(image_folder, image_file)
                #         images.append(get_image_info(image_file, self.image_min_pixel, self.image_max_pixel, self.image_resized_w, self.image_resized_h))

                # else:
                images.append(get_image_info(image_file, self.image_min_pixel, self.image_max_pixel, self.image_resized_w, self.image_resized_h))

        elif "video" in sources:
            is_video = True
            images=None
            grid_key = "video_grid_thw"
            pixel_key = "pixel_values_videos"

            video_files = sources["video"]
            video_folder = self.data_args.image_folder

            if isinstance(video_files, str):
                video_files = [video_files]

            videos = []
            for video_file in video_files:
                if not os.path.exists(video_file):
                    if not video_file.startswith("http"):
                        video_file = os.path.join(video_folder, video_file)
                video_input, video_kwargs = get_video_info(video_file, self.video_min_pixel, self.video_max_pixel, self.data_args.fps)
                videos.append(video_input)
        else:
            grid_key = None
            pixel_key = None
            images=None
            videos=None
           
        if images is None:
            
            print("No image or video found in the data.")
            images = []
            # Create a black image as a placeholder
            black_image = Image.new("RGB", (self.image_resized_w, self.image_resized_h), (0, 0, 0))
            images.append(get_image_info(black_image, self.image_min_pixel, self.image_max_pixel, self.image_resized_w, self.image_resized_h))

        elif len(images) == 0:
            print("No image or video found in the data.")
            # Create a black image as a placeholder
            black_image = Image.new("RGB", (self.image_resized_w, self.image_resized_h), (0, 0, 0))
            images.append(get_image_info(black_image, self.image_min_pixel, self.image_max_pixel, self.image_resized_w, self.image_resized_h))
        
        if videos is not None:
            
            # import ipdb; ipdb.set_trace()
            pass
            
        sources = copy.deepcopy(llava_to_openai(sources['conversations'], is_video=is_video))

        all_input_ids = [] 
        all_labels = []
        all_pixel_values = []
        all_image_grid_thw = []
        all_second_gird = []
        
        # all_dino_encoded_values = []

        # Qwen2-VL uses a default system message so I've added this.
        if len(SYSTEM_MESSAGE) > 0:
            system_message = f"{DEFAULT_IM_START_TOKEN}system\n{SYSTEM_MESSAGE}\n{DEFAULT_IM_END_TOKEN}\n"
            system_message_input_ids = processor.tokenizer(system_message, add_special_tokens=False, return_tensors='pt')['input_ids']
            system_labels = torch.full_like(system_message_input_ids, IGNORE_INDEX) 
            
            all_input_ids.append(system_message_input_ids.squeeze(0))
            all_labels.append(system_labels.squeeze(0))
            
        for _, j in enumerate(range(0, len(sources), 2)):
            
            if j >= 2:
                break
            
            user_input = sources[j]
            gpt_response = sources[j + 1]
                        
            if (DEFAULT_IMAGE_TOKEN not in user_input['content']) and (DEFAULT_VIDEO_TOKEN not in user_input['content']) and (LLAVA_IMAGE_TOKEN in user_input['content']):
                user_input = f"{DEFAULT_IM_START_TOKEN}{VISION_START_TOKEN + DEFAULT_IMAGE_TOKEN + VISION_END_TOKEN}{user_input['role']}\n{user_input['content']}\n{DEFAULT_IM_END_TOKEN}\n{DEFAULT_IM_START_TOKEN}{gpt_response['role']}\n"
                user_input = add_anchor_pad(user_input, self.anchor_token_nums, self.anchor_tokens)
                gpt_response = f"{gpt_response['content']}\n{DEFAULT_IM_END_TOKEN}\n"
                raise ValueError('Every man is a poet when he is in love')
            else:
                if self.anchor_prompt_mode != "legacy" or self.anchor_response_mode != "legacy":
                    user_input = f"{DEFAULT_IM_START_TOKEN}{user_input['role']}\n{user_input['content']}\n{DEFAULT_IM_END_TOKEN}\n{DEFAULT_IM_START_TOKEN}{gpt_response['role']}\n"
                    has_visual_token = DEFAULT_IMAGE_TOKEN in user_input or DEFAULT_VIDEO_TOKEN in user_input
                    if has_visual_token:
                        user_input = add_anchor_pad_with_mode(
                            user_input,
                            self.anchor_token_nums,
                            self.anchor_tokens,
                            self.anchor_prompt_mode,
                        )

                    response_content = gpt_response['content']
                    response_mode = (self.anchor_response_mode or "none").lower()
                    if response_mode == "cot" and has_visual_token:
                        response_content = get_comt_data_in_response(
                            response_content,
                            self.anchor_token_nums,
                            self.anchor_tokens,
                            self.anchor_task_names,
                        )
                    elif response_mode not in ("none", "answer_only", "legacy"):
                        raise ValueError(f"Unknown anchor_response_mode: {self.anchor_response_mode}")
                    gpt_response = f"{response_content}\n{DEFAULT_IM_END_TOKEN}\n"
                elif self.cur_step < self.stage_0_step:
                    user_input = f"{DEFAULT_IM_START_TOKEN}{user_input['role']}\n{user_input['content']}\n{DEFAULT_IM_END_TOKEN}\n{DEFAULT_IM_START_TOKEN}{gpt_response['role']}\n"
                    user_input = add_anchor_pad(user_input, self.anchor_token_nums, self.anchor_tokens)
                    gpt_response = f"{gpt_response['content']}\n{DEFAULT_IM_END_TOKEN}\n"
                elif self.cur_step < self.stage_1_step:
                    user_input, gpt_response = get_feature_data(user_input, gpt_response, self.anchor_token_nums, self.anchor_tokens, self.anchor_task_names)
                elif self.cur_step < self.stage_2_step:
                    user_input = f"{DEFAULT_IM_START_TOKEN}{user_input['role']}\n{user_input['content']}\n{DEFAULT_IM_END_TOKEN}\n{DEFAULT_IM_START_TOKEN}{gpt_response['role']}\n"
                    gpt_response = f"{gpt_response['content']}"
                    if DEFAULT_IMAGE_TOKEN in user_input:
                        gpt_response = get_comt_data_in_response(gpt_response, self.anchor_token_nums, self.anchor_tokens, self.anchor_task_names)
                    gpt_response = f"{gpt_response}\n{DEFAULT_IM_END_TOKEN}\n"
                    # print(f"\033[92m gpt_response: {gpt_response}\033[0m")
                else:
                    # user_input = f"{DEFAULT_IM_START_TOKEN}{user_input['role']}\n{user_input['content']}\n{DEFAULT_IM_END_TOKEN}\n{DEFAULT_IM_START_TOKEN}{gpt_response['role']}\n"
                    # gpt_response = f"{gpt_response['content']}\n{DEFAULT_IM_END_TOKEN}\n"
                    # gpt_response = replace_pad_with_anchor_tokens(gpt_response)
                    import random
                    xxx = random.randint(0, 5)
                    if xxx == 0:
                        user_input = f"{DEFAULT_IM_START_TOKEN}{user_input['role']}\n{user_input['content']}\n{DEFAULT_IM_END_TOKEN}\n{DEFAULT_IM_START_TOKEN}{gpt_response['role']}\n"
                        gpt_response = f"{gpt_response['content']}\n{DEFAULT_IM_END_TOKEN}\n"
                    else:
                        user_input = f"{DEFAULT_IM_START_TOKEN}{user_input['role']}\n{user_input['content']}\n{DEFAULT_IM_END_TOKEN}\n{DEFAULT_IM_START_TOKEN}{gpt_response['role']}\n"
                        gpt_response = f"{gpt_response['content']}"
                        if DEFAULT_IMAGE_TOKEN in user_input:
                            # INSERT_YOUR_CODE
                            total = len(self.anchor_tokens)
                            if total == 0:
                                selected_anchor_token_nums = []
                                selected_anchor_tokens = []
                                selected_anchor_task_names = []
                            else:
                                x = random.randint(1, total)
                                idxs = sorted(random.sample(range(total), x)) if x > 0 else []
                                selected_anchor_token_nums = [self.anchor_token_nums[i] for i in idxs]
                                selected_anchor_tokens = [self.anchor_tokens[i] for i in idxs]
                                selected_anchor_task_names = [self.anchor_task_names[i] for i in idxs]
                            gpt_response = get_comt_data_in_response(gpt_response, selected_anchor_token_nums, selected_anchor_tokens, selected_anchor_task_names)
                        gpt_response = f"{gpt_response}\n{DEFAULT_IM_END_TOKEN}\n"
            
            # print(f'the user_input is {user_input}')
            # print(f'the gpt_response is {gpt_response}')
                    

            # print("-----------------")
            # print(user_input, gpt_response)
            # print("-----------------")
            
            # import ipdb; ipdb.set_trace()
            
            if DEFAULT_IMAGE_TOKEN in user_input:
                inputs = processor(text=[user_input], images=images, videos=videos, padding=False, return_tensors='pt')
                prompt_input_ids = inputs['input_ids']
                # raise ValueError('Every man is a poet when he is in love')
                all_pixel_values.append(inputs[pixel_key])
                all_image_grid_thw.append(inputs[grid_key])
                
                # del dino_val
                torch.cuda.empty_cache()
                
            
            elif DEFAULT_VIDEO_TOKEN in user_input:
                if "Qwen2.5" in self.model_id:
                    inputs = processor(text=[user_input], images=images, videos=videos, padding=False, return_tensors='pt', **video_kwargs)
                    all_second_gird.extend(inputs["second_per_grid_ts"])
                else:
                    inputs = processor(text=[user_input], images=images, videos=videos, padding=False, return_tensors='pt')
                prompt_input_ids = inputs['input_ids']
                all_pixel_values.append(inputs[pixel_key])
                all_image_grid_thw.append(inputs[grid_key])

            else:
                prompt_input_ids = processor.tokenizer(user_input, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']

            response_input_ids = processor.tokenizer(gpt_response, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']

            input_ids = torch.cat([prompt_input_ids, response_input_ids], dim=1).squeeze(0)
            labels = torch.cat(
                [
                    torch.tensor([IGNORE_INDEX] * len(prompt_input_ids[0])),  
                    response_input_ids.squeeze(0),
                ],
                dim=0,
            )

            all_input_ids.append(input_ids)
            all_labels.append(labels)
        
        # There is no need for eos or bos tokens in the input_ids
        # Qwen2-VL does not use them
        input_ids = torch.cat(all_input_ids, dim=0).to(torch.long)
        labels = torch.cat(all_labels, dim=0).to(torch.long)

        # eos_token_id = processor.tokenizer.convert_tokens_to_ids(DEFAULT_IM_END_TOKEN)
        # input_ids, labels = truncate_sequence(input_ids, labels, self.max_length, eos_token_id)

        attention_mask = (input_ids > -1000000).to(torch.long)
        
        
        
        data_dict = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        if pixel_key and grid_key:
            pixel_values = torch.cat(all_pixel_values, dim=0)
            image_thw = torch.cat(all_image_grid_thw, dim=0)
            
            data_dict[pixel_key] = pixel_values
            data_dict[grid_key] = image_thw
            data_dict["image_files"] = image_files

        if len(all_second_gird) > 0:
            second_gird = all_second_gird
            data_dict["second_per_grid_ts"] = second_gird
        
        self.cur_step += 1
        
        return data_dict


class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        batch_input_ids = []
        batch_label_ids = []
        batch_pixel_values = []
        batch_pixel_video_values = []
        batch_video_thw = []
        batch_image_thw = []
        batch_second_per_grid_ts = []
        
        batch_image_files = []
        
        for example in examples:
            keys = example.keys()
            if "pixel_values_videos" in keys:
                batch_pixel_video_values.append(example["pixel_values_videos"])
                batch_video_thw.append(example["video_grid_thw"])
            elif "pixel_values" in keys:
                batch_pixel_values.append(example["pixel_values"])
                batch_image_thw.append(example["image_grid_thw"])
            
            if "image_files" in keys:
                batch_image_files.append(example["image_files"])
            
            batch_input_ids.append(example["input_ids"])
            batch_label_ids.append(example["labels"])

            if "second_per_grid_ts" in keys:
                batch_second_per_grid_ts.extend(example["second_per_grid_ts"])
        
        input_ids = pad_sequence(
            batch_input_ids, padding_side='right', padding_value=self.pad_token_id
        )

        attention_mask = input_ids != self.pad_token_id
        labels = pad_sequence(batch_label_ids, padding_side='right', padding_value=IGNORE_INDEX)

        data_dict = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
        }

        if len(batch_pixel_values) > 0:
            pixel_values = torch.cat(batch_pixel_values, dim=0)
            image_thw = torch.cat(batch_image_thw, dim=0)
            data_dict["pixel_values"] = pixel_values
            data_dict["image_grid_thw"] = image_thw

        if len(batch_pixel_video_values) > 0:
            pixel_video_values = torch.cat(batch_pixel_video_values, dim=0)
            video_thw = torch.cat(batch_video_thw, dim=0)
            data_dict["pixel_values_videos"] = pixel_video_values
            data_dict["video_grid_thw"] = video_thw

        if len(batch_second_per_grid_ts) > 0:
            data_dict["second_per_grid_ts"] = batch_second_per_grid_ts
            
        if len(batch_image_files) > 0:
            data_dict["image_files"] = batch_image_files

        return data_dict

def replace_image_tokens(input_string, is_video=False):
    if is_video:
        pattern = r'\n?' + re.escape(LLAVA_VIDEO_TOKEN) + r'\n?'
        replacement = VISION_START_TOKEN + DEFAULT_VIDEO_TOKEN + VISION_END_TOKEN
    else:
        pattern = r'\n?' + re.escape(LLAVA_IMAGE_TOKEN) + r'\n?'
        replacement = VISION_START_TOKEN + DEFAULT_IMAGE_TOKEN + VISION_END_TOKEN

    return re.sub(pattern, replacement, input_string)

def llava_to_openai(conversations, is_video=False):
    role_mapping = {"human": "user", "gpt": "assistant"}

    transformed_data = []
    for conversation in conversations:
        transformed_content = replace_image_tokens(conversation["value"], is_video=is_video)
        transformed_entry = {
            "role": role_mapping.get(conversation["from"], conversation["from"]),
            "content": transformed_content,
        }
        transformed_data.append(transformed_entry)

    return transformed_data

def make_supervised_data_module(model_id, processor, data_args, anchor_model_id):
    """Make dataset and collator for supervised fine-tuning."""
    sft_dataset = SupervisedDataset(
        data_path=data_args.data_path, processor=processor, data_args=data_args, model_id=model_id, anchor_model_id=anchor_model_id
    )
    data_collator = DataCollatorForSupervisedDataset(pad_token_id=processor.tokenizer.pad_token_id)

    return dict(train_dataset=sft_dataset,
                eval_dataset=None,
                data_collator=data_collator)
