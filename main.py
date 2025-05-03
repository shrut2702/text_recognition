from mlOCR import logger
from mlOCR.pipeline.stage_01_image_processing import ImageProcessingPipeline
from mlOCR.pipeline.stage_02_text_detection import TextDetectionPipeline
from mlOCR.pipeline.stage_03_text_recognition import TextRecognitionPipeline
from mlOCR.pipeline.stage_04_text_post_processing import TextPostProcessingPipeline


STAGE_NAME='Image Processing stage'
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = ImageProcessingPipeline()
    obj.main() #this will load image from local path but for real time inferernce we need to pass image as input
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e





STAGE_NAME='Text Detection stage'
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = TextDetectionPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e





STAGE_NAME='Text Recognition stage'
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = TextRecognitionPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e





STAGE_NAME='Text Post Processing stage'
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = TextPostProcessingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e