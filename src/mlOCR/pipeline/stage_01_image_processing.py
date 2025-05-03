from mlOCR.components.Image_Processing import ImageProcessing
from mlOCR.config.configuration import ConfigurationManager
from mlOCR.utils.common import *
from mlOCR import logger

STAGE_NAME='Image Processing stage'

class ImageProcessingPipeline:
    def __init__(self):
        pass

    def main(self, img):
        config=ConfigurationManager()
        image_processing_config=config.get_image_processing_config()
        image_processing=ImageProcessing(image_processing_config)
        image=image_processing.loadImage(img)
        save_image(image,'demo_transformed.jpg',image_processing.result_path)
        resized_image,ratio,size_heatmap=image_processing.resize_aspect_ratio(image)
        normalized_image=image_processing.normalizeMeanVariance(resized_image)
        save_image(normalized_image,'demo_normalized.jpg',image_processing.result_path)
        resized_data={'ratio':ratio,'size_heatmap':size_heatmap}
        save_bin(resized_data,'resized_data.pkl',image_processing.result_path)

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ImageProcessingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e