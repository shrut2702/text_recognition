import torch
import torch.nn as nn
from torch.autograd import Variable
from collections import OrderedDict
from skimage import io
from mlOCR.components.Image_Processing import ImageProcessing
from mlOCR.models.craft import CRAFT
from mlOCR.models.refine_net import RefineNet
from mlOCR.utils.common import *
from mlOCR.utils.text_detection_utils import *
from mlOCR import logger


class TextDetection:
    def __init__(self,TextDetectionConfig, text_threshold_arg, low_text_arg, link_threshold_arg):
        self.craft_weights=TextDetectionConfig.craft_weights
        self.refiner_weights=TextDetectionConfig.refiner_weights
        self.loaded_image_path=TextDetectionConfig.loaded_image_path
        self.normalized_image_path=TextDetectionConfig.normalized_image_path
        self.resized_data_path=TextDetectionConfig.resized_data_path
        self.text_threshold=text_threshold_arg
        self.low_text=low_text_arg
        self.link_threshold=link_threshold_arg
        # self.text_threshold=TextDetectionConfig.text_threshold
        # self.low_text=TextDetectionConfig.low_text
        # self.link_threshold=TextDetectionConfig.link_threshold
        self.poly=TextDetectionConfig.poly
        self.refine=TextDetectionConfig.refine

    def copyStateDict(self,state_dict):
        if list(state_dict.keys())[0].startswith("module"):
            start_idx = 1
        else:
            start_idx = 0
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = ".".join(k.split(".")[start_idx:])
            new_state_dict[name] = v
        return new_state_dict
    
    def modelInference(self,norm_image):
        net=CRAFT()
        net.load_state_dict(self.copyStateDict(torch.load(self.craft_weights, map_location='cpu')))
        logger.info(f'Craft model weights loaded successfully!')
        net.eval()

        x=norm_image
        x=x.astype('float32')
        x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
        x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]

        # forward pass
        with torch.no_grad():
            y, feature = net(x)

        # make score and link map
        score_text = y[0,:,:,0].cpu().data.numpy()
        score_link = y[0,:,:,1].cpu().data.numpy()

        if self.refine:
            refine_net=RefineNet()
            refine_net.load_state_dict(self.copyStateDict(torch.load(self.refiner_weights, map_location='cpu')))
            logger.info(f'RefineNet model weights loaded successfully!')
            refine_net.eval()
            self.poly=True

            with torch.no_grad():
                y_refiner = refine_net(y, feature)
            score_link = y_refiner[0,:,:,0].cpu().data.numpy()

        resized_data = load_bin(self.resized_data_path)
        ratio, size_heatmap = resized_data["ratio"], resized_data["size_heatmap"]
        ratio_h = ratio_w = 1 / ratio
        
        # Post-processing
        boxes, polys = getDetBoxes(score_text, score_link, self.text_threshold, self.link_threshold, self.low_text, self.poly)

        # coordinate adjustment
        boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = adjustResultCoordinates(polys, ratio_w, ratio_h)
        for k in range(len(polys)):
            if polys[k] is None: polys[k] = boxes[k]

        render_img = score_text.copy()
        render_img = np.hstack((render_img, score_link))
        ret_score_text = cvt2HeatmapImg(render_img)

        return boxes, polys, ret_score_text
    
    def generate_result(self):
        image=io.imread(self.loaded_image_path)
        norm_image=io.imread(self.normalized_image_path)

        boxes, polys, ret_score_text=self.modelInference(norm_image)

        filename, file_ext = os.path.splitext(os.path.basename(self.loaded_image_path))
        mask_image='artifacts/image/result/res_'+filename+'_mask.jpg'
        io.imsave(mask_image,ret_score_text)

        saveResult(self.loaded_image_path, image[:,:,::-1], polys, dirname='artifacts/image/result/')

