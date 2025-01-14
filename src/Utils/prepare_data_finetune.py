
import sys
sys.path.append("")

import os
import json
import base64
import torch
import google.generativeai as genai
from typing import List, Dict, Any, Literal
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from src.Utils.utils import no_ssl_verification, read_config
from dotenv import load_dotenv
load_dotenv()

class ImageProcessor:
    def __init__(self, config: dict, model_type: Literal['qwen2', 'gemini'] = 'qwen2'):
        """
        Initialize the ImageProcessor with specified model type and configuration.
        
        Args:
            config (dict): Configuration dictionary
            model_type (str): Type of model to use ('qwen2' or 'gemini')
        """
        self.config = config
        self.model_type = model_type.lower()
        self.dataset = []
        
        # Initialize the appropriate model
        if self.model_type == 'gemini':
            self._init_gemini()
        elif self.model_type == 'qwen2':
            self._init_qwen()
        else:
            raise ValueError("Model type must be either 'qwen2' or 'gemini'")

    def _init_gemini(self):
        """Initialize Gemini model"""
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel(
            model_name=self.config['gemini']["model_name"],
            generation_config=self.config['gemini']["generation_config"]
        )

    def _init_qwen(self):
        """Initialize qwen2 model"""
        with no_ssl_verification():
            model_id = self.config['qwen2']["model_name"]

            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2", # remove if error
                device_map="auto",
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.processor = AutoProcessor.from_pretrained(
                model_id,
                min_pixels=self.config['qwen2']["min_pixels"],
                max_pixels=self.config['qwen2']["max_pixels"],
            )

    def _load_existing_dataset(self, output_file: str):
        """Load existing dataset if it exists"""
        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as f:
                self.dataset = json.load(f)
        return {entry["messages"][1]["content"][0]["image"] for entry in self.dataset}

    def _process_gemini_image(self, image_path: str, instruction: str, system_content: str):
        """Process single image with Gemini model"""
        image_data = genai.upload_file(image_path, mime_type="image/jpeg")
        chat = self.model.start_chat(history=[])
        response = chat.send_message([image_data, instruction])
        response_text = f"```json\n{response.text}\n```"
        return response_text

    def _process_qwen_image(self, image_path: str, instruction: str, system_content: str):
        """Process single image with qwen2 model"""
        with open(image_path, "rb") as f:
            encoded_image = base64.b64encode(f.read())
        base64_data = f"data:image;base64,{encoded_image.decode('utf-8')}"

        messages = [
            {"role": "system", "content": system_content},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": base64_data},
                    {"type": "text", "text": instruction},
                ],
            },
        ]

        tokenized_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[tokenized_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        generation_config = self.model.generation_config
        generation_config.do_sample = True
        generation_config.temperature = 1.0
        generation_config.top_k = 1
        generation_config.top_p = 0.9
        generation_config.min_p = 0.1
        generation_config.best_of = 5
        generation_config.max_new_tokens = 1024
        generation_config.repetition_penalty = 1.06

        generated_ids = self.model.generate(**inputs, generation_config=generation_config)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        return output_text

    def process_images(self, output_file: str, image_paths: List[str], instruction: str, system_content: str):
        """
        Process multiple images and save results to a JSON file.
        
        Args:
            output_file (str): Path to the output JSON file
            image_paths (List[str]): List of paths to the images to process
            instruction (str): The instruction text to send to the model
            system_content (str): The system content/context
        """
        existing_images = self._load_existing_dataset(output_file)

        for image_path in image_paths:
            if image_path in existing_images:
                print(f"Image {image_path} already processed, skipping.")
                continue

            try:
                if self.model_type == 'gemini':
                    response_text = self._process_gemini_image(image_path, instruction, system_content)
                else:
                    response_text = self._process_qwen_image(image_path, instruction, system_content)

                self.dataset.append({
                    "messages": [
                        {"role": "system", "content": system_content},
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image_path},
                                {"type": "text", "text": instruction},
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": response_text},
                            ],
                        },
                    ]
                })
                print(f"Successfully processed {image_path}")

            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                continue

        # Save results
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.dataset, f, indent=4, ensure_ascii=False)

        print(f"Data has been updated in {output_file}. Processed {len(image_paths)} images.")

def get_image_paths(folder_path: str) -> List[str]:
    """Get all image paths from a folder recursively"""
    supported_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    image_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if os.path.splitext(file)[1].lower() in supported_extensions:
                image_paths.append(os.path.join(root, file))
    return image_paths


# Function to transform the data
def convert_to_sharegpt_format(input_file_path:str, output_file_path:str):
    """
    Convert data.json to sharegpt format dataset as here https://github.com/QwenLM/Qwen2-VL?tab=readme-ov-file#training
    To train with LLaMa Factory
    input_file_path: the json file created by the class ImageProcessor
    output_file_path: The sharegpt format dataset json
    """
    
    with open(input_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    transformed_data = []
    for entry in data:
        transformed_entry = {
            "messages": [],
            "images": []
        }
        for message in entry["messages"]:
            if message["role"] == "user":
                # Convert image and text content into a single message
                content_text = "<image>"
                for content in message["content"]:
                    if content["type"] == "image":
                        transformed_entry["images"].append(content["image"])
                    elif content["type"] == "text":
                        content_text += content["text"]
                transformed_entry["messages"].append({
                    "content": content_text,
                    "role": message["role"]
                })
            elif message["role"] == "assistant":
                for content in message["content"]:
                    if content["type"] == "text":
                        transformed_entry["messages"].append({
                            "content": content["text"],
                            "role": message["role"]
                        })
        transformed_data.append(transformed_entry)

    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(transformed_data, file, ensure_ascii=False, indent=2)
    
    print(f"Transformed data has been saved to {output_file_path}")

    return transformed_data


if __name__ == "__main__":
    # Load configuration
    config = read_config("config/training_config.yaml")
    
    # Read system instruction and question from config
    with open(config['system_instruction_path'], 'r') as f:
        system_content = f.read().strip()
    
    with open(config['question_path'], 'r') as f:
        question = f.read().strip()

    output_file = "data234.json"
    train_folder = "train_data"
    
    # Get image paths
    image_paths = get_image_paths(train_folder)[:2]
    print(f"Found {len(image_paths)} images to process")

    
    # Process with qwen2
    print("\nProcessing with qwen2 model...")
    qwen_processor = ImageProcessor(config, model_type='qwen2')
    qwen_processor.process_images(output_file, image_paths, question, system_content)

    image_paths = get_image_paths(train_folder)[3:5]
    print(f"Found {len(image_paths)} images to process")

    # Process with Gemini
    print("\nProcessing with Gemini model...")
    gemini_processor = ImageProcessor(config, model_type='gemini')
    gemini_processor.process_images(output_file, image_paths, question, system_content)

    # Load the input JSON file
    input_file_path = 'data234.json'
    output_file_path = 'golf_data234.json'
    convert_to_sharegpt_format(input_file_path, output_file_path)
