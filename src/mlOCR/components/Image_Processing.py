import numpy as np
from skimage import io
import cv2

class ImageProcessing:
    def __init__(self,ImageProcessingConfig):
        self.image_path=ImageProcessingConfig.image_path
        self.result_path=ImageProcessingConfig.result_path
        self.canvas_size=ImageProcessingConfig.canvas_size
        self.mag_ratio=ImageProcessingConfig.mag_ratio
        self.norm_mean=ImageProcessingConfig.norm_mean
        self.norm_variance=ImageProcessingConfig.norm_variance

        self.norm_mean=np.array([self.norm_mean[0],self.norm_mean[1],self.norm_mean[2]],dtype=np.float32)
        self.norm_variance=np.array([self.norm_variance[0],self.norm_variance[1],self.norm_variance[2]],dtype=np.float32)

        
    def loadImage(self):
        img = io.imread(self.image_path)           # RGB order
        if img.shape[0] == 2: img = img[0]
        if len(img.shape) == 2 : img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if img.shape[2] == 4:   img = img[:,:,:3]
        img = np.array(img)

        return img
    
    def normalizeMeanVariance(self,in_img):
        # should be RGB order
        img = in_img.copy().astype(np.float32)

        norm_mean_array = self.norm_mean * 255.0
        norm_variance_array = self.norm_variance * 255.0

        img -= norm_mean_array
        img /= norm_variance_array
        return img

    def denormalizeMeanVariance(self,in_img):
        # should be RGB order
        img = in_img.copy()
        img *= self.norm_variance
        img += self.norm_mean
        img *= 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img
    
    def resize_aspect_ratio(self,img, interpolation=cv2.INTER_LINEAR):
        mag_ratio=self.mag_ratio
        square_size=self.canvas_size

        height, width, channel = img.shape

        # magnify image size
        target_size = mag_ratio * max(height, width)

        # set original image size
        if target_size > square_size:
            target_size = square_size
        
        ratio = target_size / max(height, width)    

        target_h, target_w = int(height * ratio), int(width * ratio)
        proc = cv2.resize(img, (target_w, target_h), interpolation = interpolation)


        # make canvas and paste image
        target_h32, target_w32 = target_h, target_w
        if target_h % 32 != 0:
            target_h32 = target_h + (32 - target_h % 32)
        if target_w % 32 != 0:
            target_w32 = target_w + (32 - target_w % 32)
        resized = np.zeros((target_h32, target_w32, channel), dtype=np.float32)
        resized[0:target_h, 0:target_w, :] = proc
        target_h, target_w = target_h32, target_w32

        size_heatmap = (int(target_w/2), int(target_h/2))

        return resized, ratio, size_heatmap
    
    def cvt2HeatmapImg(self,img):
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        return img