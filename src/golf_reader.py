
import sys
sys.path.append("")

from typing import Dict, Any, List
from src.Utils.utils import read_config

class BaseGolfReader:
    def __init__(self, config_path: str):
        self.config = self._read_config(config_path)
        self._initialize()

    def _read_config(self, config_path: str):
        # Assuming read_config is defined elsewhere
        return read_config(config_path)

    def _initialize(self):
        """Initialize model-specific components. To be implemented by child classes."""
        raise NotImplementedError

    def _extract_json(self, text: str) -> Dict[str, Any]:
        try:
            start_index = text.find('{')
            end_index = text.rfind('}') + 1
            json_string = text[start_index:end_index]
            json_string = json_string.replace('true', 'True').replace('false', 'False').replace('null', 'None')
            return eval(json_string)
        except Exception as e:
            print(f"Error parsing JSON: {e}")
            return {}

    def transform_scores(self, data):
        try:
            for player in data.get('golf', []):
                player['score'] = {f"hole_{i+1}": score for i, score in enumerate(player['score'])}
            return data
        except Exception as e:
            print(f'Error transforming scores: {e}')
            return data

    def process_image(self, image_path: str) -> Dict[str, Any]:
        """Process a single image. To be implemented by child classes."""
        raise NotImplementedError


