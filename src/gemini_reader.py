import sys
sys.path.append("")

from typing import Dict, Any
from src.golf_reader import BaseGolfReader
from dotenv import load_dotenv
import google.generativeai as genai
import os


class GeminiReader(BaseGolfReader):
    def _initialize(self):
        load_dotenv()
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = self._create_model()

    def _create_model(self):
        with open(self.config['system_instruction_path'], 'r') as f:
            system_instruction = f.read()
        return genai.GenerativeModel(
            model_name=self.config['model_name'],
            generation_config=self.config['generation_config'],
            system_instruction=system_instruction,
        )

    def _upload_to_gemini(self, path: str, mime_type: str = None):
        file = genai.upload_file(path, mime_type=mime_type)
        print(f"Uploaded file '{file.display_name}' as: {file.uri}")
        return file

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
            
            with open(self.config['question_path'], 'r') as f:
                question = f.read()
            response = chat_session.send_message(question)
            raw_data = self._extract_json(response.text)
            
            return self.transform_scores(raw_data)

        except Exception as e:
            print(f"Error processing image: {e}")
            return None
        
if __name__ == "__main__":
    import json
    from pprint import pprint

    config_path = "config/config.yaml"

    gemini_reader = GeminiReader(config_path)
    print("\nGemini Single image processing:")
    result = gemini_reader.process_image("images/2.png")
    pprint(json.dumps(result, indent=4, ensure_ascii=False))
