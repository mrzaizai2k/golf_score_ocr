
import sys
sys.path.append("")

from typing import Dict, Any, List
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from Utils.vision_util import process_vision_info
from src.golf_reader import BaseGolfReader

class Qwen2Reader(BaseGolfReader):
    def _initialize(self):
        self.model_name = self.config['qwen2']['model_name']
        self.model = self._create_model()
        self.processor = self._create_processor()
        
        with open(self.config['question_path'], 'r') as f:
            self.question = f.read().strip()

    def _create_model(self):
        return Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )

    def _create_processor(self):
        qwen_config = self.config.get('qwen2', {})
        min_pixels = qwen_config.get('min_pixels', 200_704)
        max_pixels = qwen_config.get('max_pixels', 1_003_520)
        return AutoProcessor.from_pretrained(
            self.model_name,
            padding_side="left",
            min_pixels=min_pixels,
            max_pixels=max_pixels
        )

    def process_images(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        messages = [
            [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": self.question}
                ]
            }]
            for image_path in image_paths
        ]

        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        results = []
        for output_text in output_texts:
            raw_data = self._extract_json(output_text)
            golf_data = self.transform_scores(raw_data)
            results.append(golf_data)

        return results

    def process_image(self, image_path: str) -> Dict[str, Any]:
        results = self.process_images([image_path])
        return results[0] if results else {}
    
if __name__ == "__main__":
    import json
    from pprint import pprint

    config_path = "config/config.yaml"
    qwen_reader = Qwen2Reader(config_path)

    test_images = ["images/2.png", "images/4.png"]
    results = qwen_reader.process_images(test_images)
    print("\nBatch processing results:")
    for image_path, result in zip(test_images, results):
        pprint(json.dumps(result, indent=4, ensure_ascii=False))

    print("\nSingle image processing:")
    result = qwen_reader.process_image("images/2.png")
    pprint(json.dumps(result, indent=4, ensure_ascii=False))


