from mlOCR.constants import *
from mlOCR.utils.common import read_yaml, create_directories
from mlOCR.entity.config_entity import CraftModelConfig, ImageProcessingConfig,DemoImageConfig

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
    
    def get_craft_model_config(self)->CraftModelConfig:
        config=self.config.craft_model

        craft_model_config=CraftModelConfig(
            model_weight=config.model_path
        )

        return craft_model_config
