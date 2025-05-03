""" from mlOCR import logger
from mlOCR.pipeline.stage_01_image_processing import ImageProcessingPipeline
from mlOCR.pipeline.stage_02_text_detection import TextDetectionPipeline
from mlOCR.pipeline.stage_03_text_recognition import TextRecognitionPipeline
from mlOCR.pipeline.stage_04_text_post_processing import TextPostProcessingPipeline

STAGE_NAME='Inference stage'

class InferencePipeline:
    def __init__(self):
        pass

    def inference(self, img, text_threshold_arg=0.7, low_text_arg=0.4, link_threshold_arg=0.4):
        obj = ImageProcessingPipeline()
        obj.main(img)
        
        obj = TextDetectionPipeline()
        obj.main(text_threshold_arg, low_text_arg, link_threshold_arg)
        
        obj = TextRecognitionPipeline()
        obj.main()
        
        obj = TextPostProcessingPipeline()
        output=obj.main()

        return output


# just for showing how to call the pipeline in a single script
if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = InferencePipeline()
        output=obj.inference(img='demo.jpg')
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
         """

from mlOCR import logger
from mlOCR.pipeline.stage_01_image_processing import ImageProcessingPipeline
from mlOCR.pipeline.stage_02_text_detection import TextDetectionPipeline
from mlOCR.pipeline.stage_03_text_recognition import TextRecognitionPipeline
from mlOCR.pipeline.stage_04_text_post_processing import TextPostProcessingPipeline

STAGE_NAME='Inference stage'

class InferencePipeline:
    def __init__(self):
        pass

    def inference(self, img, text_type, text_threshold_arg=0.7, low_text_arg=0.4, link_threshold_arg=0.4):
        try:
            yield "data: Processing the Image...\n\n"
            obj = ImageProcessingPipeline()
            obj.main(img)
            
            yield "data: Extracting Text from the Image...\n\n"
            obj = TextDetectionPipeline()
            obj.main(text_threshold_arg, low_text_arg, link_threshold_arg)
            
            yield "data: Recognizing the Text...\n\n"
            obj = TextRecognitionPipeline(text_type)
            obj.main()
            
            yield "data: Post Processing the Detected Text...\n\n"
            obj = TextPostProcessingPipeline()
            output=obj.main()

            yield f"data: DONE::{output}\n\n"

        except Exception as e:
            logger.exception(f"Error in inference pipeline: {e}")
            yield f"data: ERROR::{str(e)}\n\n"




# just for showing how to call the pipeline in a single script
if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = InferencePipeline()
        output=obj.inference(img='demo.jpg')
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
        