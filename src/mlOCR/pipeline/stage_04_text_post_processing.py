from mlOCR.components.Text_Post_Processing import TextPostProcessing
from mlOCR.config.configuration import ConfigurationManager
from mlOCR.utils.common import *
from mlOCR import logger

STAGE_NAME='Text Post Processing stage'

class TextPostProcessingPipeline:
    def __init__(self):
        pass

    def main(self):
        config_manager=ConfigurationManager()
        text_post_processing_config=config_manager.get_text_post_processing_config()
        text_post_processing=TextPostProcessing(text_post_processing_config)
        text=text_post_processing.get_text()
        corrected_text=text_post_processing.correct_spelling(text=text)
        return corrected_text

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = TextPostProcessingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e