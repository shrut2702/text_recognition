import cv2
import numpy as np
import os
import tensorflow as tf
import keras
from keras import layers
from keras import ops
from mlOCR.models.crnn_pred import crnn_pred_model
from mlOCR.utils.common import save_text_file

class TextRecognition:
    def __init__(self, TextRecognitionConfig, text_type):
        self.crnn_weights_digital=TextRecognitionConfig.crnn_weights_digital
        self.crnn_weights_handwritten=TextRecognitionConfig.crnn_weights_handwritten
        self.crnn_input_path=TextRecognitionConfig.crnn_input_path
        self.crnn_output_path=TextRecognitionConfig.crnn_output_path
        self.resize_canvas=TextRecognitionConfig.resize_canvas
        self.sorting_threshold=TextRecognitionConfig.sorting_threshold
        self.char_list=TextRecognitionConfig.char_list
        self.char_to_num = layers.StringLookup(vocabulary=list(self.char_list), mask_token=None, oov_token=None)
        self.num_to_char = layers.StringLookup(vocabulary=self.char_list, invert=True, mask_token=None, oov_token="*")
        self.text_type=text_type
        if self.text_type=='digital':
            self.crnn_weights=self.crnn_weights_digital
        elif self.text_type=='handwritten':
            self.crnn_weights=self.crnn_weights_handwritten
    
    def get_image(self,img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        return img

    def get_word_coordinates(self,coord_path):
        word_coordinates=[]
        with open(coord_path,'r') as file:
            lines=file.readlines()

            for line in lines:
                line=line.replace('\n','')
                if line!='':
                    coord=line.split(',')
                    coord=list(map(int,coord))
                    word_coordinates.append(coord)
        
        word_coordinates = np.array(word_coordinates)

        # Check if there are any coordinates to process
        if len(word_coordinates) == 0:
            return []

        boxes = sorted(word_coordinates, key=lambda x: x[1])  # x[1] is the top-left y-coordinate

        # Group boxes into lines based on y-distance threshold
        lines = []
        current_line = [boxes[0]]
        
        threshold=self.sorting_threshold
        # threshold=0.5
        for i in range(1, len(boxes)):
            max_height_index=0
            max_height=0
            for j in range(len(current_line)):
                word_height=abs(max(np.array(current_line[j])[[1, 3, 5, 7]]) - min(np.array(current_line[j])[[1, 3, 5, 7]]))
                if word_height>max_height:
                    max_height=word_height
                    max_height_index=j

            if abs(boxes[i][1] - current_line[max_height_index][1]) < threshold*max_height:  # Adjust threshold as needed
                current_line.append(boxes[i])
            else:
                lines.append(sorted(current_line, key=lambda x: x[0]))  # Sort by x-coordinates (left to right)
                current_line = [boxes[i]]
            
        
        lines.append(sorted(current_line, key=lambda x: x[0]))

        # print(len(lines))
        # for line in lines:
        #     print(f'{line}\n')

        # Flatten the list of sorted lines
        sorted_boxes = [box for line in lines for box in line]
        
        return sorted_boxes

    def get_each_word_image(self,img:np.ndarray,coord:list):
        x1,y1,x2,y2,x3,y3,x4,y4=coord
        
        x_start=min(x1,x4)
        y_start=min(y1,y2)

        x_end=max(x2,x3)
        y_end=max(y3,y4)

        cropped_img=img[y_start:y_end+1,x_start:x_end+1]

        return cropped_img


    def image_transformation(self,img):
        _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        white = np.sum(thresh == 255)
        black = np.sum(thresh == 0)

        if black > white:
            img = cv2.bitwise_not(img)

        permuted_canvas_size = self.resize_canvas[::-1]  #(h,w) to (w,h)
        img = cv2.resize(img, permuted_canvas_size, interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=-1)

        img = np.transpose(img, axes=[1, 0, 2])

        return img

    def cropped_images(self):
        img=self.get_image(os.path.join(self.crnn_input_path,'demo_transformed.jpg'))
        word_coordinates=self.get_word_coordinates(os.path.join(self.crnn_input_path,'res_demo_transformed.txt'))

        cropped_images=[]
        for coord in word_coordinates:
            
            cropped_img=self.get_each_word_image(img,coord)
            if cropped_img is not None and cropped_img.size > 0:
                transformed_img=self.image_transformation(cropped_img)
            
                cropped_images.append(transformed_img)

        return cropped_images
    
    def ctc_decode(self, y_pred, input_length, greedy=True, beam_width=100, top_paths=1):
        input_shape = ops.shape(y_pred)
        num_samples, num_steps = input_shape[0], input_shape[1]
        y_pred = ops.log(ops.transpose(y_pred, axes=[1, 0, 2]) + keras.backend.epsilon())
        input_length = ops.cast(input_length, dtype="int32")

        if greedy:
            (decoded, log_prob) = tf.nn.ctc_greedy_decoder(
                inputs=y_pred, sequence_length=input_length
            )
        else:
            (decoded, log_prob) = tf.compat.v1.nn.ctc_beam_search_decoder(
                inputs=y_pred,
                sequence_length=input_length,
                beam_width=beam_width,
                top_paths=top_paths,
            )
        decoded_dense = []
        for st in decoded:
            st = tf.SparseTensor(st.indices, st.values, (num_samples, num_steps))
            decoded_dense.append(tf.sparse.to_dense(sp_input=st, default_value=-1))
        return (decoded_dense, log_prob)
    
    def decode_batch_predictions(self,pred):
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        # Use greedy search. For complex tasks, you can use beam search
        results = self.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
            :, :
        ]
        # Iterate over the results and get back the text
        output_text = []
        for res in results:
            res = tf.strings.reduce_join(self.num_to_char(res)).numpy().decode("utf-8")
            res=res.replace("*","")
            output_text.append(res)
        return output_text
    
    def get_predictions(self, cropped_images: list):
        crnn_pred=crnn_pred_model()
        crnn_pred.load_weights(self.crnn_weights)
        if len(cropped_images)>0:
            predictions=crnn_pred.predict(np.array(cropped_images))
            decoded_predictions=self.decode_batch_predictions(predictions)
            text=' '.join(decoded_predictions)
        else:
            text=''

        return text
    
    def save_results(self, text: str):
        save_text_file(text=text, filename='crnn_raw_text.txt', path=self.crnn_output_path)