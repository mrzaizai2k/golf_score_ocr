import sys
sys.path.append("")

from paddleocr import PaddleOCR
from PIL import Image
import numpy as np

from src.Utils.utils import read_config, resize_same_ratio, rotate_image

class OcrReader:
    def __init__(self,  
                 config_path:str = "config/config.yaml",
                 logger=None
                 ):
        
        self.config_path = config_path
        self.config = read_config(path = self.config_path)['ocr']
        self.resize_size = self.config['resize_size']
        self.logger = logger


    def get_image(self, input_data:any) -> Image:
        if isinstance(input_data, str):  # If input_data is a path
            image = Image.open(input_data)
        elif isinstance(input_data, Image.Image):  # If input_data is a PIL image
            image = input_data
        else:
            raise ValueError("Unsupported input data type")
        
        # image = resize_same_ratio(image, target_size=self.resize_size)
        return image


    def get_text(self, input_data) -> dict:
        # Detect the language of the image
        try:
            image = self.get_image(input_data)
            image, doc_angle = rotate_image(image)

            ocr = PaddleOCR(lang="vi", 
                            show_log=False, 
                            use_angle_cls=True, 
                            cls=True,
                            use_gpu=False) # ocr_version='PP-OCR2', enable_mkldnn=True) #https://github.com/PaddlePaddle/PaddleOCR/issues/11597
 
            result = ocr.ocr(np.array(image))

            # Combine the recognized text from the OCR result
            text = " ".join([line[1][0] for line in result[0]])
            # If translation is not required, use the original text and language
            data = {
                "ori_text": text,
            }
            data['angle'] = doc_angle

            if self.logger:
                self.logger.debug(f"ocr_data: {data}")

        except Exception as e:
            print("error", e)
            data = {
                    "ori_text": "",
                    "angle": 0,
                }
            if self.logger:
                self.logger.debug(f"error: {e}")
        return data
    
if __name__ == "__main__":

    config_path = "config/config.yaml"
    img_path = "images/1_1.png"
    ocr_reader = OcrReader(config_path=config_path, )
    text = ocr_reader.get_text(input_data=img_path)
    print('text', text)