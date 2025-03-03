from mlOCR.config.configuration import ConfigurationManager
from mlOCR.components.Demo_Image import DemoImage
from mlOCR import logger

STAGE_NAME='Demo Image import stage'

class DemoImagePipeline:
    def __int__(self):
        pass

    def main(self):
        config=ConfigurationManager()
        demo_image_config=config.get_demo_image_config()
        demo_image=DemoImage(demo_image_config)
        demo_image.download_image()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DemoImagePipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e