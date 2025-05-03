from mlOCR.components.Text_Detection import TextDetection
from mlOCR.config.configuration import ConfigurationManager
from mlOCR.utils.common import *
from mlOCR import logger

STAGE_NAME='Text Detection stage'

class TextDetectionPipeline:
    def __init__(self):
        pass

    def main(self,  text_threshold_arg, low_text_arg, link_threshold_arg):
        config=ConfigurationManager()
        text_detection_config=config.get_text_detection_config()
        text_detection=TextDetection(text_detection_config, text_threshold_arg, low_text_arg, link_threshold_arg)
        text_detection.generate_result()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = TextDetectionPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e