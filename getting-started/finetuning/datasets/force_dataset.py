# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 3 Community License Agreement.


import copy
import itertools

import torch
from datasets import load_dataset
from PIL import Image 
import json


# check system prompt token seq or user prompt token seq is in the current token list
def check_header(targets, seq):
    for i in range(len(seq) - 3):
        if seq[i : i + 3] in targets:
            return True
    return False


def replace_target(target, seq):
    for i in range(len(seq) - 3):
        if seq[i : i + 3] == target:
            seq[i], seq[i + 1], seq[i + 2] = -100, -100, -100
    return seq


def tokenize_dialogs(dialogs, images, processor):
    text_prompt = processor.apply_chat_template(dialogs)
    text_prompt = [prompt.replace('<|begin_of_text|>','') for prompt in text_prompt]
    batch = processor(
        images=images,
        text=text_prompt,
        padding=True,
        return_tensors="pt",
    )
    label_list = []
    for i in range(len(batch["input_ids"])):
        dialog_tokens = batch["input_ids"][i].tolist()
        labels = copy.copy(dialog_tokens)
        eot_indices = [i for i, n in enumerate(labels) if n == 128009]
        last_idx = 0
        # system prompt header "<|start_header_id|>system<|end_header_id|>" has been tokenized to [128006, 9125, 128007]
        # user prompt header "<|start_header_id|>user<|end_header_id|>" has been tokenized to [128006, 882, 128007]
        prompt_header_seqs = [[128006, 9125, 128007], [128006, 882, 128007]]
        for n, idx in enumerate(eot_indices):
            current_seq = labels[last_idx : idx + 1]
            if check_header(prompt_header_seqs, current_seq):
                # found prompt header, indicating that this seq should be masked
                labels[last_idx : idx + 1] = [-100] * (idx - last_idx + 1)
            else:
                last_idx = idx + 1
            #  Mask all the assistant header prompt <|start_header_id|>assistant<|end_header_id|>, which has been tokenized to [128006, 78191, 128007]
        assistant_header_seq = [128006, 78191, 128007]
        labels = replace_target(assistant_header_seq, labels)
        # Mask the padding token and image token 128256
        for i in range(len(labels)):
            if (
                labels[i] == processor.tokenizer.pad_token_id or labels[i] == 128256
            ):  #  128256 is image token index
                labels[i] = -100
        label_list.append(labels)
    batch["labels"] = torch.tensor(label_list)
    return batch



def get_custom_dataset(dataset_config, processor, split, split_ratio=0.9):
    data_files = {"train": "/home/choiyj/remote_data/dataset_train.csv", "test": "/home/choiyj/remote_data/dataset_test.csv", "validation": "/home/choiyj/remote_data/dataset_validation.csv"}
    # load_dataset will return DatasetDict that contains all the data in the train set
    dataset_dict = load_dataset("csv", data_files=data_files)
    dataset = dataset_dict["train"]
    dataset = dataset.select(range(10000))
    dataset = dataset.train_test_split(
        test_size=1 - split_ratio, shuffle=True, seed=42
    )[split]
    return dataset


def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


class ForceDataCollator:
    def __init__(self, processor):
        self.processor = processor
        self.processor.tokenizer.padding_side = (
            "right"  # during training, one always uses padding on the right
        )

    def __call__(self, samples):
        dialogs, images = [], []
        for sample in samples:
            color_image_path, ext_color_image_path, object_name, force_data, gripper_data, robot_pos_data, robot_cur_data = sample["color_image"], sample["ext_color_image"], sample["object_name"], sample["force_data"], sample["gripper_data"], sample["robot_pos_data"], sample["robot_cur_data"]
            
            color_image = Image.open(color_image_path)
            ext_color_image = Image.open(ext_color_image_path)

            color_image = color_image.convert("RGB")
            ext_color_image = ext_color_image.convert("RGB")

            gripper_data = json.loads(gripper_data)

            gripper_pos = str(gripper_data['gripper_pos'])
            gripper_cur = str(gripper_data['gripper_cur'])

            concat_color_image = get_concat_v(color_image, ext_color_image)
            question = f"At a specific moment during the process of grasping, lifting, or releasing the object, what are the forces (sensor1_x, sensor1_y, sensor1_z, sensor2_x, sensor2_y, sensor2_z) exerted by both fingers of the gripper? The gripper opening is {gripper_pos}. The end-effector pose of the robotic arm at this instant is defined by its position and orientation {robot_pos_data}. The joint current of the robotic arm at this instant is {robot_cur_data}."

            dialog = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": force_data,
                        }
                    ],
                },
            ]

            dialogs.append(dialog)
            images.append([concat_color_image])
        return tokenize_dialogs(dialogs, images, self.processor)


def get_data_collator(processor):
    return ForceDataCollator(processor)
