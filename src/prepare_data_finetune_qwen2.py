
import sys
sys.path.append("")

import os
import json
import base64
import torch
import google.generativeai as genai
from typing import List, Dict, Any
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from src.Utils.utils import no_ssl_verification, read_config
from dotenv import load_dotenv
load_dotenv()

def update_finetuning_results_gemini(config:dict, output_file: str, image_paths: List[str], instruction: str, system_content: str):
    """
    Process golf scorecard images using Gemini model and save results to a JSON file.
    
    Args:
        output_file (str): Path to the output JSON file
        image_paths (List[str]): List of paths to the images to process
        instruction (str): The instruction text to send to the model
        system_content (str): The system content/context
        api_key (str): Gemini API key
    """
    # Configure Gemini
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel(
        model_name=config["model_name"],
        generation_config=config["generation_config"]
    )

    # Load existing dataset if file exists
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    else:
        dataset = []

    # Create a lookup for existing entries
    existing_images = {entry["messages"][1]["content"][0]["image"] for entry in dataset}

    # Process each image path
    for image_path in image_paths:
        if image_path in existing_images:
            print(f"Image {image_path} already processed, skipping.")
            continue

        try:
            # Upload image to Gemini
            image_data = genai.upload_file(image_path, mime_type="image/jpeg")
            
            # Create chat session
            chat = model.start_chat(history=[])
            
            # Send message with image and instruction
            response = chat.send_message([
                image_data,
                instruction
            ])
            
            # Extract the JSON part from the response
            response_text = response.text
            response_text = f"```json\n{response_text}\n```"


            # Add result to dataset
            dataset.append({
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

    # Write updated dataset to JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)

    print(f"Data has been updated in {output_file}. Processed {len(image_paths)} images.")


def update_finetuning_results(output_file, model_path, image_paths, instruction, system_content):
    # Load model, tokenizer, and processor
    with no_ssl_verification():
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        processor = AutoProcessor.from_pretrained(
            model_path,
            min_pixels=224 * 224,
            max_pixels=2048 * 2048,
        )

    # Load existing dataset if file exists
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    else:
        dataset = []

    # Create a lookup for existing entries
    existing_images = {entry["messages"][1]["content"][0]["image"] for entry in dataset}

    # Process each image path
    for image_path in image_paths:
        if image_path in existing_images:
            print(f"Image {image_path} already processed, skipping.")
            continue

        # Encode image as base64
        with open(image_path, "rb") as f:
            encoded_image = base64.b64encode(f.read())
        decoded_image_text = encoded_image.decode('utf-8')
        base64_data = f"data:image;base64,{decoded_image_text}"

        # Create message structure
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

        # Prepare prompt
        tokenized_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[tokenized_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Generate output
        generation_config = model.generation_config
        generation_config.do_sample = True
        generation_config.temperature = 1.0
        generation_config.top_k = 1
        generation_config.top_p = 0.9
        generation_config.min_p = 0.1
        generation_config.best_of = 5
        generation_config.max_new_tokens = 2048
        generation_config.repetition_penalty = 1.06

        generated_ids = model.generate(**inputs, generation_config=generation_config)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # Add result to dataset
        dataset.append({
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
                        {"type": "text", "text": output_text[0]},
                    ],
                },
            ]
        })

    # Write updated dataset to JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)

    print(f"Data has been updated in {output_file}. Processed {len(image_paths)} images.")


def get_image_paths(folder_path):
    """
    Recursively retrieve all image file paths from a folder, including its subfolders.
    Args:
        folder_path (str): The path to the folder to search for images.
    Returns:
        list: A list of image file paths.
    """
    supported_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    image_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if os.path.splitext(file)[1].lower() in supported_extensions:
                image_paths.append(os.path.join(root, file))
    return image_paths

if __name__ == "__main__":
    # Configuration
    model_path = "erax/EraX-VL-7B-V1"
    output_file = "train_data/data.json"
    system_content = (
        "You are a helpful assistant that responds in JSON format with the golf score card information. "
        "Do not add any annotations there. Remember to close any bracket."
    )
    instruction = (
        "From this image, return json that has field list is the list of information (hand written text) that each element has field name is the name of player, "
        "score is the list of all scores of that player in format like this. The golf score is calculated like this, they hasve 9 front and 9 back scores. "
        "Remember that the each score would be from 0-9 but after 9 digits, which mean the 10th element is the sum of the last 9 scores, and the 20th element will be the sum of hole 11 to 19. "
        "for example the score will be [0, 1, 2, 1, 3, 2, 0, 0, 1, 11]"
        """[{"golf": [{"name": "", "score": []}, {"name": "", "score": []}]"""
        "These are examples: ```json\n[\n    {\n        \"golf\": [\n            {\"name\": \"T Anh\", \"score\": [2, 3, 1, 1, 2, 1, 1, 0, 3, 14]},\n            {\"name\": \"Tấn\", \"score\": [1, 0, 1, 1, 0, 1, 2, 2, 2, 10]},\n            {\"name\": \"A Phúc\", \"score\": [1, 0, 2, 1, 1, 2, 2, 2, 2, 13]},\n            {\"name\": \"A Tuấn\", \"score\": [2, 0, 1, 0, 0, 4, 3, 1, 4, 15]}\n        ]\n    }\n]\n```"
        "or ```json\n[\n    {\n        \"golf\": [\n            {\"name\": \"C Ngân\", \"score\": [2, 4, 3, 2, 2, 1, 1, 1, 2, 18]},\n            {\"name\": \"C Ngân\", \"score\": [2, 2, 1, 4, 0, 2, 2, 2, 2, 17]}\n        ]\n    }\n]\n```"
    )

    train_folder = "train_data"
    image_paths = get_image_paths(train_folder)
    print(image_paths)


    # Example usage
    # image_paths = ["train_data/score_card_2/IMG_9376.jpg", "train_data/score_card_2/IMG_9378.JPG"]
    # update_finetuning_results(output_file, model_path, image_paths, instruction, system_content)

    config_path = "config/config.yaml"
    config = read_config(config_path)

    update_finetuning_results_gemini(config=config, output_file=output_file, image_paths=image_paths, instruction=instruction, system_content=system_content)