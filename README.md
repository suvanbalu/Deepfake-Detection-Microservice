# Deepfake Detection Solution

## Overview

This project aims to develop a robust and easily deployable solution for deepfake detection, addressing the challenges posed by the evolving nature of deepfake technology. The implemented solution includes a detection model, a Flask API, and containerization for efficient deployment.

## Key Components

### Detection Model

The detection model is based on the EfficientB0 architecture and has been trained using 4000 fake images and 1201 real images. The training process involved 35 epochs to ensure a comprehensive learning experience.

### Flask API

The Flask API, implemented in the `app.py` file, serves as the interface for interacting with the deepfake detection model. It allows seamless integration into different applications, providing a user-friendly experience.

You can start the API by running the following command:

```bash
pip install -r requirements.txt
python app.py
```

You can use the `postman-collection.json` file to test the API using Postman or Thunder Client.

### Docker Container

Yet to be implemented.

## Additional Information

### Face Extraction

The MTCNN (Multi-Task Cascaded Convolutional Networks) algorithm is used for face extraction as part of the preprocessing steps.

### Model Training

Various experiments were conducted during the model training phase to ensure optimal performance. The chosen model, trained with the specified dataset and parameters, demonstrates effectiveness in detecting deepfake content. ()

## Stats

Metric | Value
--- | ---
Total Models Trained Successfully | 26
Total Time Taken for Model Training | 2261 minutes ~37 hrs
Average Time Taken for Model Training | 87 minutes ~1.45 hrs
Max Time Taken to Train a Model | 191 minutes ~3.18 hrs
Min Number of Images Used for Training | 1601
Max Number of Images Used for Training | 7201
Min Number of Iterations for a Model to Train | 15
Max Number of Iterations for a Model to Train | 50
| Total faces extracted | 11011 |
| Total time taken | 513 minutes ~ 8.55 hrs |
| Total videos processed | ~3700 |

## Contribution

Contributions to the project are welcome. If you encounter any issues or have suggestions for improvement, feel free to submit a pull request.