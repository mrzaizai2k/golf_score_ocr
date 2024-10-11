
import sys
sys.path.append("")

import os
import google.generativeai as genai
from typing import Dict, Any
from src.Utils.utils import read_config

from dotenv import load_dotenv
load_dotenv()
class GolfReader:
    def __init__(self, config_path: str):
        load_dotenv()
        self.config = read_config(config_path)
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = self._create_model()

    def _create_model(self):
        system_instruction_path=self.config['system_instruction_path']
        with open(system_instruction_path, 'r') as f:
            system_instruction = f.read() 
        return genai.GenerativeModel(
            model_name=self.config['model_name'],
            generation_config=self.config['generation_config'],
            system_instruction = system_instruction,
        )

    def _upload_to_gemini(self, path: str, mime_type: str = None):
        file = genai.upload_file(path, mime_type=mime_type)
        print(f"Uploaded file '{file.display_name}' as: {file.uri}")
        return file

    def _extract_json(self, text: str) -> Dict[str, Any]:
        print(text)
        start_index = text.find('{')
        end_index = text.rfind('}') + 1
        json_string = text[start_index:end_index]
        json_string = json_string.replace('true', 'True').replace('false', 'False').replace('null', 'None')
        return eval(json_string)

    def process_image(self, image_path: str) -> Dict[str, Any]:
        try:
            uploaded_file = self._upload_to_gemini(image_path, mime_type="image/png")
            
            chat_session = self.model.start_chat(
                history=[
                    {
                        "role": "user",
                        "parts": [uploaded_file],
                    },
                ]
            )
            
            question_path = self.config['question_path']
            with open(question_path, 'r') as f:
                question = f.read()
            response = chat_session.send_message(question)
            raw_data = self._extract_json(response.text)
            
            golf_data = self.transform_scores(raw_data)
            
        
            return golf_data

        except Exception as e:
            print(f"Error processing image: {e}")
            return None 

    def transform_scores(self, data):
        try:
            for player in data['golf']:
                player['score'] = {f"hole_{i+1}": score for i, score in enumerate(player['score'])}
            return data
            
        except Exception as e:
            print(f'error: {e}')
            return data


# Usage example:
if __name__ == "__main__":
    import json
    config_path = "config/config.yaml"
    golf_reader = GolfReader(config_path)
    result = golf_reader.process_image("images/2.png")
    print('result', result)
    print(json.dumps(result, indent=4,  ensure_ascii=False))
