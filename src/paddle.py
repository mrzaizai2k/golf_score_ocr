# from paddleocr import PaddleOCR
# from transformers import pipeline
# from PIL import Image

# class OcrReader:
#     def __init__(self,  
#                  translator=None,
#                  config_path:str = "config/config.yaml",
#                  logger=None
#                  ):
        
#         self.config_path = config_path
#         self.config = read_config(path = self.config_path)['ocr']

#         self.language_dict_path = self.config['language_dict_path']
#         self.language_detector = self.config['language_detector']
#         self.language_thresh = self.config['language_thresh']
#         self.target_language = self.config['target_language']
#         self.resize_size = self.config['resize_size']

#         self.logger = logger

#         # Load language dictionary from JSON file
#         with open(self.language_dict_path, 'r', encoding='utf-8') as f:
#             self.language_dict = json.load(f)

#         device = self.config.get('device', None)
#         # Set up the device (CPU or GPU)
#         if device:
#             self.device = device
#         else:
#             self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
#         print("Device", self.device)

#         self.translator = translator

#         # Load zero-shot image classification model
#         self.initialize_language_detector()

    
#     def initialize_language_detector(self):
#         # Create a dummy image for model initialization
#         dummy_image = Image.new("RGB", (224, 224), color=(255, 255, 255))  # White image
#         candidate_labels = ["en", "fr"]  # Example labels

#         # Initialize the zero-shot image classification model
#         self.image_classifier = pipeline(task="zero-shot-image-classification", 
#                                          model=self.language_detector, 
#                                          device=self.device,
#                                          batch_size=8)

#         # Perform a dummy inference to warm up the model
#         self.image_classifier(dummy_image, candidate_labels=candidate_labels)
#         print("Model pipeline initialized with dummy data.")

#     def get_image(self, input_data:any) -> Image:
#         if isinstance(input_data, str):  # If input_data is a path
#             image = Image.open(input_data)
#         elif isinstance(input_data, Image.Image):  # If input_data is a PIL image
#             image = input_data
#         else:
#             raise ValueError("Unsupported input data type")
        
#         image = resize_same_ratio(image, target_size=self.resize_size)
#         return image
    
#     def _get_lang(self, image: Image.Image) -> str:
#         # Define candidate labels for language classification
#         candidate_labels = [f"language {key}" for key in self.language_dict]

#         # Perform inference to classify the language
#         outputs = self.image_classifier(image, candidate_labels=candidate_labels)
#         outputs = [{"score": round(output["score"], 4), "label": output["label"] } for output in outputs]
        
#         # Extract the language with the highest score
#         language_names = [entry['label'].replace('language ', '') for entry in outputs]
#         scores = [entry['score'] for entry in outputs]
#         abbreviations = [self.language_dict.get(language) for language in language_names]
        
#         first_abbreviation = abbreviations[0]
#         lang = 'en'  # Default to English
        
#         if scores[0] > self.language_thresh:
#             lang = first_abbreviation

#         return lang

#     def get_text(self, input_data) -> dict:
#         # Detect the language of the image
#         try:
#             image = self.get_image(input_data)
#             image, doc_angle = rotate_image(image)
            
#             src_language = self._get_lang(image)

#             if self.logger:
#                 self.logger.debug(f"src_language: {src_language}")

#             # Initialize the PaddleOCR with the detected language
#             ocr = None

#             if (src_language in ["zh-CN","ch", "chinese_cht", "japan"]) and (self.device == 'cpu'):
#                 print("src_language", src_language)
#                 print("self.device", self.device)
#                 ocr = PaddleOCR(lang="en", show_log=False, use_angle_cls=True, 
#                                 cls=True,) # ocr_version='PP-OCR2', enable_mkldnn=True) #https://github.com/PaddlePaddle/PaddleOCR/issues/11597
#             else:
#                 ocr = PaddleOCR(lang=src_language, show_log=False, use_angle_cls=True, cls=True, )
            

#             result = ocr.ocr(np.array(image))

#             # Combine the recognized text from the OCR result
#             text = " ".join([line[1][0] for line in result[0]])

#             # Handle translation if a translator and target language are provided
#             if self.translator and self.target_language:
#                 trans_text, src_language = self.translator.translate(text=text, to_lang=self.target_language)

#                 data = {
#                     "ori_text": text,
#                     "ori_language": src_language,
#                     "text": trans_text,
#                     "language": self.target_language,
#                 }
#             else:
#                 # If translation is not required, use the original text and language
#                 trans_text, src_language = text, src_language
#                 data = {
#                     "ori_text": text,
#                     "ori_language": src_language,
#                     "text": trans_text,
#                     "language": src_language,
#                 }
#             data['angle'] = doc_angle

#             if self.logger:
#                 self.logger.debug(f"ocr_data: {data}")

#         except Exception as e:
#             print("error", e)
#             data = {
#                     "ori_text": "",
#                     "ori_language": "",
#                     "text": "",
#                     "language": "",
#                     "angle": 0,
#                 }
#             if self.logger:
#                 self.logger.debug(f"error: {e}")
#         return data
    
