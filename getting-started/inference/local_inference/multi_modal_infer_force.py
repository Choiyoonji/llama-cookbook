import argparse
import os
import sys

import gradio as gr
import torch
from accelerate import Accelerator
from huggingface_hub import HfFolder
from peft import PeftModel
from PIL import Image as PIL_Image
from datasets import load_dataset
from PIL import Image 
import json
from transformers import MllamaForConditionalGeneration, MllamaProcessor

# Initialize accelerator
accelerator = Accelerator()
device = accelerator.device

# Constants
DEFAULT_MODEL = "meta-llama/Llama-3.2-11B-Vision-Instruct"
FINETUNING_PATH = "/home/choiyj/llama_ws/PEFT/model_force_0320_2"
MAX_OUTPUT_TOKENS = 2048
MAX_IMAGE_SIZE = (1120, 1120)


def get_hf_token():
    """Retrieve Hugging Face token from the cache or environment."""
    # Check if a token is explicitly set in the environment
    token = os.getenv("HUGGINGFACE_TOKEN")
    if token:
        return token

    # Automatically retrieve the token from the Hugging Face cache (set via huggingface-cli login)
    token = HfFolder.get_token()
    if token:
        return token

    print("Hugging Face token not found. Please login using `huggingface-cli login`.")
    sys.exit(1)


def load_model_and_processor(model_name: str, finetuning_path: str = None):
    """Load model and processor with optional LoRA adapter"""
    print(f"Loading model: {model_name}")
    hf_token = get_hf_token()
    model = MllamaForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        device_map=device,
        token=hf_token,
    )
    processor = MllamaProcessor.from_pretrained(
        model_name, token=hf_token, use_safetensors=True
    )
    finetuning_path = FINETUNING_PATH
    if finetuning_path and os.path.exists(finetuning_path):
        print(f"Loading LoRA adapter from '{finetuning_path}'...")
        model = PeftModel.from_pretrained(
            model, finetuning_path, is_adapter=True, torch_dtype=torch.bfloat16
        )
        print("LoRA adapter merged successfully")
    else:
        print("wrong path")

    model, processor = accelerator.prepare(model, processor)
    return model, processor


def process_image(image_path: str = None, image=None) -> PIL_Image.Image:
    """Process and validate image input"""
    if image is not None:
        return image.convert("RGB")
    if image_path and os.path.exists(image_path):
        return PIL_Image.open(image_path).convert("RGB")
    raise ValueError("No valid image provided")


def generate_text_from_image(
    model, processor, image, prompt_text: str, temperature: float, top_p: float
):
    """Generate text from image using model"""
    conversation = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt_text}],
        }
    ]
    prompt = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    inputs = processor(
        image, prompt, text_kwargs={"add_special_tokens": False}, return_tensors="pt"
    ).to(device)
    print("Input Prompt:\n", processor.tokenizer.decode(inputs.input_ids[0]))
    output = model.generate(
        **inputs, temperature=temperature, top_p=top_p, max_new_tokens=MAX_OUTPUT_TOKENS
    )
    return processor.decode(output[0])[len(prompt) :]

def get_custom_dataset():
    # load_dataset will return DatasetDict that contains all the data in the train set
    dataset_dict = load_dataset("csv", data_files="/home/choiyj/remote_data/dataset_2.csv")
    dataset = dataset_dict["train"]
    return dataset

def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

def main(args):
    """Main execution flow"""
    model, processor = load_model_and_processor(
        args.model_name, args.finetuning_path
    )

    test_data = get_custom_dataset()
    results = []

    for sample in test_data:
        color_image_path, ext_color_image_path, object_name, force_data, gripper_data, robot_pos_data, robot_cur_data = sample["color_image"], sample["ext_color_image"], sample["object_name"], sample["force_data"], sample["gripper_data"], sample["robot_pos_data"], sample["robot_cur_data"]
        
        timestep = sample['frame_id']

        color_image = Image.open(color_image_path)
        ext_color_image = Image.open(ext_color_image_path)

        color_image = color_image.convert("RGB")
        ext_color_image = ext_color_image.convert("RGB")

        gripper_data = json.loads(gripper_data)

        gripper_pos = str(gripper_data['gripper_pos'])

        concat_color_image = get_concat_v(color_image, ext_color_image)
        question = f"At a specific moment during the process of grasping, lifting, or releasing the object, what are the forces (sensor1_x, sensor1_y, sensor1_z, sensor2_x, sensor2_y, sensor2_z) exerted by both fingers of the gripper? The object is a {object_name}. The gripper opening is {gripper_pos}. The end-effector pose of the robotic arm at this instant is defined by its position and orientation {robot_pos_data}. The joint current of the robotic arm at this instant is {robot_cur_data}."

    
        image = process_image(image=concat_color_image)
        result = generate_text_from_image(
            model, processor, image, question, args.temperature, args.top_p
        )
        print("Generated Text:", result)

        results.append({
            "frame_id": timestep,
            "generated_text": result
        })

    # 결과를 json 파일로 저장
    output_path = "generated_results_2.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-modal inference with optional Gradio UI and LoRA support"
    )
    parser.add_argument("--image_path", type=str, help="Path to the input image")
    parser.add_argument("--prompt_text", type=str, help="Prompt text for the image")
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature"
    )
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument(
        "--model_name", type=str, default=DEFAULT_MODEL, help="Model name"
    )
    parser.add_argument("--finetuning_path", type=str, help="Path to LoRA weights")
    parser.add_argument("--gradio_ui", action="store_true", help="Launch Gradio UI")

    args = parser.parse_args()
    main(args)