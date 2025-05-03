from mlOCR.components.Text_Recognition import TextRecognition
from mlOCR.config.configuration import ConfigurationManager
from mlOCR.utils.common import *
from mlOCR import logger

STAGE_NAME='Text Recognition stage'

class TextRecognitionPipeline:
    def __init__(self):
        pass

    def main(self, text_type):
        config_manager=ConfigurationManager()
        text_recognition_config=config_manager.get_text_recognition_config()
        text_recognition=TextRecognition(text_recognition_config, text_type)
        cropped_images=text_recognition.cropped_images()
        text=text_recognition.get_predictions(cropped_images=cropped_images)
        text_recognition.save_results(text=text)

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = TextRecognitionPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e