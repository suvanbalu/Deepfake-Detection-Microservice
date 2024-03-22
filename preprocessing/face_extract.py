import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import cv2
import random
from mtcnn.mtcnn import MTCNN
from cv2 import dnn
import json
import logging
import tqdm
import glob
import logging

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils import load_metadata
from logs.enablelogging import setup_logging, close_logging

prototxt = r'models\face_detection_models\deploy.prototxt.txt'  
caffemodel=r'models\face_detection_models\res10_300x300_ssd_iter_140000.caffemodel'

def load_config(config_path):
    try:
        with open(config_path) as config_file:
            return json.load(config_file)
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_path}")
        exit(1)


def extract_frames_uniform(video_path, interval_seconds):
    try:
        capture = cv2.VideoCapture(video_path)
        frame_rate = capture.get(cv2.CAP_PROP_FPS)
        interval_frames = int(frame_rate * interval_seconds)

        frames = []
        frame_idx = 0
        while True:
            ret, frame = capture.read()
            if not ret:
                break
            if frame_idx % interval_frames == 0:
                frames.append(frame)
            frame_idx += 1
        capture.release()
        return frames
    except Exception as e:
        logging.error(f"Error extracting frames from video: {e}")
        return None


def extract_frames_rand(video_path, num_frames_to_select):
    try:
        capture = cv2.VideoCapture(video_path)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        if num_frames_to_select > total_frames:
            logging.warning(f"Requested number of frames ({num_frames_to_select}) exceeds total frames in video ({total_frames}). Selecting all frames.")
            num_frames_to_select = total_frames

        selected_frame_indices = random.sample(range(total_frames), min(num_frames_to_select, total_frames))
        selected_frames = []

        for frame_idx in selected_frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = capture.read()
            if ret:
                selected_frames.append(frame)

        capture.release()
        return selected_frames
    except Exception as e:
        logging.error(f"Error extracting frames from video: {e}")
        return None

def load_face_detection_model(method, model_paths=None):
    if method == 'MTCNN':
        return MTCNN()
    elif method == 'SSD':
        if not model_paths:
            logging.error("Model paths required for SSD face detection.")
            exit(1)
        prototxt = model_paths['prototxt']
        caffemodel = model_paths['caffemodel']
        return cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
    else:
        logging.error(f"Unsupported face detection method: {method}")
        exit(1)

# @tf.function(reduce_retracing=True)
def extract_face(frame, method, margin, detector=None, net=None):
    if not detector and method == 'MTCNN':
        detector = load_face_detection_model(method)
    if not net and method == 'SSD':
        model_paths = {'prototxt': prototxt, 'caffemodel': caffemodel}  # Assuming these are defined globally
        net = load_face_detection_model(method, model_paths)

    if method == 'MTCNN':
        try:
            faces = detector.detect_faces(frame)
            if faces:
                x, y, w, h = faces[0]['box']
                x -= margin
                y -= margin
                w += 2 * margin
                h += 2 * margin
                # Ensure the coordinates are within image bounds
                x = max(0, x)
                y = max(0, y)
                face = frame[y:y + h, x:x + w]
                return face
            else:
                return None
        except Exception as e:
            logging.error(f"Error detecting face using MTCNN: {e}")
            return None
    elif method == 'SSD':
        try:
            resized_frame = cv2.resize(frame, (300, 300))
            blob = dnn.blobFromImage(resized_frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
            net.setInput(blob)
            detections = net.forward()
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]]
                    x1, y1, x2, y2 = box.astype(int)
                    # Add margin to the bounding box
                    x1 -= margin
                    y1 -= margin
                    x2 += margin
                    y2 += margin
                    # Ensure the coordinates are within image bounds
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(frame.shape[1], x2)
                    y2 = min(frame.shape[0], y2)
                    face = frame[y1:y2, x1:x2]
                    return face
            return None
        except Exception as e:
            logging.error(f"Error detecting face using SSD: {e}")
            return None
    else:
        logging.error(f"Unsupported face detection method: {method}")
        exit(1)


def save_faces(frames, output_dir, filename, method, margin, resize=0):
    for i, frame in enumerate(frames):
        face = extract_face(frame, method, margin)
        if face is not None:
            if resize!=0:
                face = cv2.resize(face, (resize, resize))
            cv2.imwrite(os.path.join(output_dir, f"{filename}_{i}.jpg"), face)
            logging.info(f"Saved face from frame {i} of {filename}")


def process_video(video_path, output_dir, config):
    dir = os.path.dirname(video_path)
    filename = os.path.basename(video_path)
    logging.info(f"Processing video: {filename}")
    try:
        meta_flag=config.get("meta_flag",True)
        if meta_flag:
            metadata = load_metadata(dir,metadata_type=config['metadata_type'])
            label = metadata[filename]
            output_dir = os.path.join(output_dir,label)
        
        if config['extract_method'] == 'uniform':
            frames = extract_frames_uniform(video_path, config['interval_seconds'])
        elif config['extract_method'] == 'random':
            frames = extract_frames_rand(video_path, config['num_frames_to_select'])
        else:
            logging.error(f"Unsupported frame extraction method: {config['extract_method']}")
            exit(1)

        save_faces(frames, output_dir, filename, config['face_detection']['method'], config["face_detection"]['margin'], config.get('resize'))
        logging.info(f"Finished processing video: {os.path.join(output_dir,filename)}")
    except Exception as e:
        logging.error(f"Error processing video: {e}")

def process_image(image_path, output_dir, config):
    filename = os.path.basename(image_path)
    logging.info(f"Processing image: {filename}")
    try:
        frame = cv2.imread(image_path)
        save_faces([frame], output_dir, filename, config['face_detection']['method'], config["face_detection"]['margin'], config.get('resize'))
        logging.info(f"Finished processing image: {os.path.join(output_dir,filename)}")
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        

def main(config_path, log_dir="logs/face_extract"):
    try:
        setup_logging(log_dir)
        logging.info(f"Using configuration: {config_path}")
    except Exception as e:
        print("Error setting up logging: ", e)
        exit(1)
    try:
        config = load_config(config_path)
        dataset_dirs = config['dataset_dir']
        # print(dataset_dirs)
        output_dirs = config['output_dir']
        if len(output_dirs)==1:
            output_dirs = [output_dirs[0] for _ in range(len(dataset_dirs))]
        if config["media_type"]=="video":
            for dataset_dir, output_dir in zip(dataset_dirs, output_dirs):
                if config.get("meta_flag",True):
                    os.makedirs(os.path.join(output_dir,"REAL"), exist_ok=True)
                    os.makedirs(os.path.join(output_dir,"FAKE"), exist_ok=True)
                else:
                    os.makedirs(output_dir, exist_ok=True)
                logging.info(f"Processing dataset: {dataset_dir}")
                for video in glob.glob(os.path.join(dataset_dir, '*.mp4')):
                    process_video(video, output_dir, config)
        elif config["media_type"]=="image":
            for dataset_dir, output_dir in zip(dataset_dirs, output_dirs):
                if config.get("meta_flag",True):
                    os.makedirs(os.path.join(output_dir,"REAL"), exist_ok=True)
                    os.makedirs(os.path.join(output_dir,"FAKE"), exist_ok=True)
                else:
                    os.makedirs(output_dir, exist_ok=True)
                logging.info(f"Processing dataset: {dataset_dir}")
                for image in glob.glob(os.path.join(dataset_dir, '*.jpg')):
                    print(image)
                    process_image(image, output_dir, config)
                
    except Exception as e:
        print("An error occurred: ", e)
        logging.error(f"An error occurred: {e}")
    finally:
        close_logging()

if __name__ == "__main__":
    main(r"config/face_extraction/linux_face_extract.json")
