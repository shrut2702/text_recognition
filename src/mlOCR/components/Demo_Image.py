import requests
import os
from mlOCR import logger
from mlOCR.entity.config_entity import DemoImageConfig

class DemoImage:
    def __init__(self,DemoImageConfig):
        self.url=DemoImageConfig.image_url
        self.image_dir=DemoImageConfig.image_dir

    def download_image(self):
        filename='demo.jpg'
        if not os.path.exists(os.path.join(self.image_dir, filename)):
            response=requests.get(self.url)

            if response.status_code == 200:
                with open(os.path.join(self.image_dir, filename), "wb") as file:
                    file.write(response.content)
                logger.info(f'{filename} downloaded successfully!')
            else:
                logger.info(f'Error downloading {filename}!')
        else:
            logger.info(f'{filename} already exists!')