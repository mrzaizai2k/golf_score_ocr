
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from pprint import pprint
import torch 

model_dir = "models/qwen2/qwen2_vl_lora_sft_2b"
# default: Load the model on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_dir, device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
)      

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2-VL-2B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processer
# processor = AutoProcessor.from_pretrained(model_dir, padding_side="left")

# The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
min_pixels = 256*28*28
max_pixels = 1280*28*28
processor = AutoProcessor.from_pretrained(model_dir, min_pixels=min_pixels, max_pixels=max_pixels, padding_side="left")

messages1 = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "train_data/score_card_2/IMG_9377.JPG",
            },
            {"type": "text", "text": "From this image, return json that has field list is the list of information (hand written text) that each element has field name is the name of player, score is the list of all scores of that player in format like this. The golf score is calculated like this, they hasve 9 front and 9 back scores. Remember that the each score would be from 0-9 but after 9 digits, which mean the 10th element is the sum of the last 9 scores, and the 20th element will be the sum of hole 11 to 19. for example the score will be [0, 1, 2, 1, 3, 2, 0, 0, 1, 11][{\"golf\": [{\"name\": \"\", \"score\": []}, {\"name\": \"\", \"score\": []}]"},
        ],
    }
]
messages2 = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "train_data/score_card_2/IMG_9377.JPG",
            },
            {"type": "text", "text": "From this image, return json that has field list is the list of information (hand written text) that each element has field name is the name of player, score is the list of all scores of that player in format like this. The golf score is calculated like this, they hasve 9 front and 9 back scores. Remember that the each score would be from 0-9 but after 9 digits, which mean the 10th element is the sum of the last 9 scores, and the 20th element will be the sum of hole 11 to 19. for example the score will be [0, 1, 2, 1, 3, 2, 0, 0, 1, 11][{\"golf\": [{\"name\": \"\", \"score\": []}, {\"name\": \"\", \"score\": []}]"},
        ],
    }
]


messages = [messages1, messages2] 
texts = [
    processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    for msg in messages
]

image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=texts,
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=512)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
pprint(output_text)
