from mlOCR.constants import *
from mlOCR.utils.common import read_yaml, create_directories
from mlOCR.entity.config_entity import (TextPostProcessingConfig, TextRecognitionConfig, TextDetectionConfig,
                                         ImageProcessingConfig,DemoImageConfig)
from pathlib import Path

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH,
        schema_filepath = SCHEMA_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])

    def get_demo_image_config(self)->DemoImageConfig:
        config=self.config.demo_image

        create_directories([config.image_dir])

        demo_image_config=DemoImageConfig(
            image_url=config.image_url,
            image_dir=config.image_dir
        )

        return demo_image_config

    def get_image_processing_config(self)->ImageProcessingConfig:
        config=self.config.image_processing
        params=self.params.craft_image_processing

        create_directories([config.result_path])

        image_processing_config=ImageProcessingConfig(
            image_path=Path(config.image_path),
            result_path=Path(config.result_path),
            norm_mean=tuple(params.norm_mean.translate(str.maketrans('','','()')).split(',')),
            norm_variance=tuple(params.norm_variance.translate(str.maketrans('','','()')).split(',')),
            canvas_size=int(params.canvas_size),
            mag_ratio=float(params.mag_ratio)
        )

        return image_processing_config
    
    def get_text_detection_config(self)->TextDetectionConfig:
        config=self.config.text_detection
        params=self.params.text_detection

        text_detection_config=TextDetectionConfig(
            craft_weights=Path(config.craft_weights),
            refiner_weights=Path(config.refiner_weights),
            loaded_image_path=Path(config.loaded_image_path),
            normalized_image_path=Path(config.normalized_image_path),
            resized_data_path=Path(config.resized_data_path),
            text_threshold=float(params.text_threshold),
            low_text=float(params.low_text),
            link_threshold=float(params.link_threshold),
            poly=bool(params.poly),
            refine=bool(params.refine)
        )

        return text_detection_config
    
    def get_text_recognition_config(self)->TextRecognitionConfig:
        config=self.config.text_recognition
        params=self.params.text_recognition

        create_directories([config.crnn_output_path])

        text_recognition_config=TextRecognitionConfig(
            crnn_weights_digital=Path(config.crnn_weights_digital),
            crnn_weights_handwritten=Path(config.crnn_weights_handwritten),
            crnn_input_path=Path(config.crnn_input_path),
            crnn_output_path=Path(config.crnn_output_path),
            resize_canvas=tuple(map(int,params.resize_canvas.translate(str.maketrans('','','()')).split(','))),
            sorting_threshold=float(params.sorting_threshold),
            char_list=params.char_list
        )

        return text_recognition_config
    
    def get_text_post_processing_config(self)->TextPostProcessingConfig:
        config=self.config.text_post_processing

        text_post_processing_config=TextPostProcessingConfig(
            text_input_path=Path(config.text_input_path)
        )

        return text_post_processing_config
