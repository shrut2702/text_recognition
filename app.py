from flask import Flask, request, jsonify, Response
from mlOCR.pipeline.inference_pipeline import InferencePipeline
from mlOCR import logger
from flask_cors import CORS
from PIL import Image, UnidentifiedImageError
import io
import numpy as np

app = Flask(__name__)
CORS(app) 

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

""" @app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.content_length and request.content_length > 25 * 1024 * 1024:
            return "data: ERROR::File too large (max 25MB)\n\n", 400

        if 'image' not in request.files:
            return jsonify({'error': 'No image file in the request'}), 400
        
        image_file = request.files['image']
        if not allowed_file(image_file.filename):
            return jsonify({'error': 'Invalid file type'}), 400

        image = Image.open(io.BytesIO(image_file.read())).convert("RGB")

        # Convert to numpy array if your OCR pipeline expects arrays
        image_np = np.array(image)
        
        if image_file is None and image_file.size == 0:
            return jsonify({'error': 'Empty image file'}), 400
        
        try:
            text_threshold = float(request.form.get('text_threshold', 0.7))
            size = float(request.form.get('size', 0.4))  
            proximity = float(request.form.get('proximity', 0.4))

            if not (0 <= text_threshold <= 1):
                return "data: ERROR::Invalid text_threshold value\n\n", 400
            if not (0 <= size <= 1):
                return "data: ERROR::Invalid size value\n\n", 400
            if not (0 <= proximity <= 1):
                return "data: ERROR::Invalid proximity value\n\n", 400
        except ValueError:
            return "data: ERROR::Parameters must be numbers\n\n", 400


        # Run inference
        STAGE_NAME = 'Inference stage'
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        pipeline = InferencePipeline()
        output = pipeline.inference(
            img=image_np,
            text_threshold_arg=text_threshold,
            low_text_arg=proximity,
            link_threshold_arg=size
        )
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")

        return jsonify({"text": output})

    except Exception as e:
        logger.exception(e)
        return jsonify({"error": str(e)}), 500 """
    

@app.route('/predict-stream', methods=['POST'])
def predict_stream():
    try:
        if request.content_length and request.content_length > 25 * 1024 * 1024:
            return "data: ERROR::File too large (max 25MB)\n\n", 400
        
        if 'image' not in request.files:
            return "data: ERROR::No image file\n\n", 400

        image_file = request.files['image']
        if not allowed_file(image_file.filename):
            return "data: ERROR::Invalid file type\n\n", 400

        try:
            image = Image.open(io.BytesIO(image_file.read())).convert("RGB")
        except UnidentifiedImageError:
            return "data: ERROR::Unrecognized or corrupted image\n\n", 400
        image_np = np.array(image)
        if image_file is None and image_file.size == 0:
            return "data: ERROR::Empty image file\n\n", 400

        text_type = request.form.get('text_type', 'digital')

        try:
            text_threshold = float(request.form.get('text_threshold', 0.7))
            size = float(request.form.get('size', 0.4))  
            proximity = float(request.form.get('proximity', 0.4))

            if not (0 <= text_threshold <= 1):
                return "data: ERROR::Invalid text_threshold value\n\n", 400
            if not (0 <= size <= 1):
                return "data: ERROR::Invalid size value\n\n", 400
            if not (0 <= proximity <= 1):
                return "data: ERROR::Invalid proximity value\n\n", 400
        except ValueError:
            return "data: ERROR::Parameters must be numbers\n\n", 400

        return Response(
            InferencePipeline().inference(image_np, text_type, text_threshold, proximity, size),
            mimetype='text/event-stream'
        )
    except Exception as e:
        return Response(f"data: ERROR::{str(e)}\n\n", mimetype='text/event-stream')


if __name__ == '__main__':
    # app.run(host="0.0.0.0", port=5000, debug=True)
    app.run(host="0.0.0.0", port=8080)