from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import cv2
import random
import numpy as np
import logging
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from mtcnn.mtcnn import MTCNN
from cv2 import dnn

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

model = load_model('models/deepfake_detection_models/faces_trained_model.h5')

prototxt = 'models/face_detection_models/deploy.prototxt.txt'  
caffemodel = 'models/face_detection_models/res10_300x300_ssd_iter_140000.caffemodel'

def load_face_detection_model(method, model_paths=None):
    if method == 'MTCNN':
        return MTCNN()
    elif method == 'SSD':
        net = dnn.readNetFromCaffe(model_paths['prototxt'], model_paths['caffemodel'])
        return net
    else:
        logging.error("Unsupported model for face detection.")
        return None

def extract_frames_rand(video_path, num_frames_to_select):
    capture = cv2.VideoCapture(video_path)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    selected_frame_indices = random.sample(range(total_frames), num_frames_to_select)
    selected_frames = []

    for frame_idx in selected_frame_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = capture.read()
        if ret:
            selected_frames.append(frame)

    capture.release()
    return selected_frames

def extract_face(frame, method='MTCNN', margin=0):
    detector = None
    net = None
    if method == 'MTCNN':
        detector = load_face_detection_model(method)
    elif method == 'SSD':
        model_paths = {'prototxt': prototxt, 'caffemodel': caffemodel}
        net = load_face_detection_model(method, model_paths)
    else:
        return None

    if method == 'MTCNN' and detector:
        faces = detector.detect_faces(frame)
        if faces:
            x, y, w, h = faces[0]['box']
            x -= margin
            y -= margin
            w += 2 * margin
            h += 2 * margin
            x, y, w, h = max(0, x), max(0, y), w, h
            face = frame[y:y + h, x:x + w]
            return cv2.resize(face, (224, 224))
        return None
    elif method == 'SSD' and net:
        (h, w) = frame.shape[:2]
        blob = dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                startX, startY = max(0, startX - margin), max(0, startY - margin)
                endX, endY = min(w, endX + margin), min(h, endY + margin)
                face = frame[startY:endY, startX:endX]
                return cv2.resize(face, (224, 224))
        return None
    else:
        return None

def predict_from_frames(frames, face_detection_method):
    faces = [extract_face(frame, method=face_detection_method) for frame in frames]
    faces = [face for face in faces if face is not None]
    if not faces:
        return [], []
    faces = np.array(faces)
    faces = preprocess_input(faces)
    predictions = model.predict(faces)
    return predictions, predictions

def predict_from_video(video_path, num_frames=3, face_detection_method='MTCNN'):
    if not os.path.exists(video_path):
        return {"error": "Video Not Found"}, []

    frames = extract_frames_rand(video_path, num_frames)
    predictions, probs = predict_from_frames(frames, face_detection_method)
    if len(predictions)==0:
    
        return {"error": "No Faces Detected, Try Changing the num_frame"}, []
    return format_predictions(predictions, probs)

def predict_from_image(image_path, face_detection=True, face_detection_method='MTCNN'):
    if not os.path.exists(image_path):
        return {"error": "Image Not Found"}, []

    frame = cv2.imread(image_path)
    if face_detection:
        face = extract_face(frame, method=face_detection_method)
        if face is None:
            return {"error": "No Face Detected"}, []
        face = np.expand_dims(face, axis=0)
        print(face.shape)
        frame = preprocess_input(face)
    else:
        frame = cv2.resize(frame, (224, 224))
        frame = preprocess_input(frame)
        frame = np.expand_dims(frame, axis=0)
    predictions = model.predict(frame)
    return format_predictions(predictions, predictions)

def format_predictions(predictions, probs):
    formatted_predictions = []
    for prediction, prob in zip(predictions, probs):
        label = "Fake" if prediction > 0.5 else "Real"
        probability = prob if label == "Fake" else 1 - prob
        formatted_predictions.append({"prediction": label, "probability": float(probability)})
    return formatted_predictions

@app.route('/', methods=['GET'])
def home():
    return "Welcome to Deepfake Detection API"

@app.route('/detect_deepfake_video', methods=['POST'])
def detect_deepfake_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files['video']
    video_path = 'uploaded_video.mp4'
    video_file.save(video_path)

    face_detection_method = request.form.get('face_detection_method', 'MTCNN')
    num_frames = int(request.form.get('num_frames', 3))
    predictions = predict_from_video(video_path, num_frames=num_frames, face_detection_method=face_detection_method)

    os.remove(video_path)
    return jsonify({"predictions":predictions,"face_detection_method":face_detection_method,"num_frames":num_frames}), 200

@app.route('/detect_deepfake_image', methods=['POST'])
def detect_deepfake_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']
    image_path = 'uploaded_image.jpg'
    image_file.save(image_path)

    face_detection = request.form.get('face_detection', '1') == '1'
    face_detection_method = request.form.get('face_detection_method', 'MTCNN')
    # print(face_detection_method)
    predictions = predict_from_image(image_path, face_detection, face_detection_method)

    os.remove(image_path)
    return jsonify({"predictions":predictions,"face_detection_method":face_detection_method}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=True, host='0.0.0.0', port=port)
