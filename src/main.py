
import sys
sys.path.append("")

import os
import google.generativeai as genai
from src.Utils.utils import read_config

from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

config_path = "config/config.yaml" 
config = read_config(config_path)

def upload_to_gemini(path, mime_type=None):
  """Uploads the given file to Gemini.

  See https://ai.google.dev/gemini-api/docs/prompting_with_media
  """
  file = genai.upload_file(path, mime_type=mime_type)
  print(f"Uploaded file '{file.display_name}' as: {file.uri}")
  return file

# Create the model


generation_config = config['generation_config']

model = genai.GenerativeModel(
  model_name=config['model_name'],
  generation_config=generation_config,
  # safety_settings = Adjust safety settings
  # See https://ai.google.dev/gemini-api/docs/safety-settings
  system_instruction=config['system_instruction'],
)

# TODO Make these files available on the local file system
# You may need to update the file paths
files = [
  upload_to_gemini("images/1.png", mime_type="image/png"),
]

chat_session = model.start_chat(
  history=[
    {
      "role": "user",
      "parts": [
        files[0],
      ],
    },
  ]
)

response = chat_session.send_message(config['question'])

print(response.text)

def extract_json(text: str) -> dict:
    start_index = text.find('{')
    end_index = text.rfind('}') + 1
    json_string = text[start_index:end_index]
    json_string = json_string.replace('true', 'True').replace('false', 'False').replace('null', 'None')
    result = eval(json_string)
    return result

golf_data = {"golf": []}
import json 
data =  extract_json(response.text)
for name, scores in data.items():
    # Create the score dictionary for each person
    score_dict = {
        f"hole_{i+1}": score for i, score in enumerate(scores)
    }
    
    # Create a JSON structure with name and score
    golf_data["golf"].append({
        "name": name,
        "score": score_dict
    })

# Output the transformed JSON structure
print(json.dumps(golf_data, indent=4))