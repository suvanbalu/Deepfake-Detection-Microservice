# Deepfake Detection as a Microservice

## Introduction

In an era where digital content is ubiquitous, the proliferation of deepfake technology poses unprecedented challenges to the integrity of information online. Deepfakes—synthetic media in which a person's likeness has been manipulated to create convincing, yet entirely fictional content—have the potential to mislead, defraud, and undermine public discourse. Recognizing the critical need for robust countermeasures, our project, "Deepfake Detection as a Microservice," aims to provide a powerful, accessible solution to this modern dilemma.

This repository houses a cutting-edge microservice designed to identify and flag deepfake content with precision and speed. Leveraging advanced machine learning and image processing techniques, our service is engineered to integrate seamlessly with various platforms, offering a vital tool for content creators, social media platforms, and news organizations to ensure the authenticity of their media.

### What This Repository Contains

- **Source Code**: The core algorithms and server code that power the deepfake detection service.
- **Documentation**: Detailed documentation on how to install, configure, and utilize the service, including API references.
- **Examples**: Sample requests and responses, showcasing the microservice in action.
- **Tests**: A suite of automated tests to ensure the service's reliability and accuracy.

### Features

- **High Accuracy**: Utilizes state-of-the-art deep learning models trained on extensive datasets to detect deepfakes with high precision.
- **Easy Integration**: Designed as a microservice, it can be easily incorporated into existing digital platforms via a straightforward API.
- **Scalability**: Engineered to handle requests at scale, ensuring reliable performance even under heavy load.
- **Continuous Learning**: Regularly updated models to adapt to the evolving techniques used in deepfake generation.

### Use Cases

- **Social Media Platforms**: Automatically scan and flag uploaded videos and images for deepfake content.
- **News Organizations**: Verify the authenticity of media before publication or broadcast.
- **Content Creators**: Ensure the integrity of content shared with audiences.

Stay tuned as we continue to develop and refine this crucial technology in the fight against digital deception. Our commitment is to provide an accessible, effective tool to safeguard digital content authenticity, empowering users worldwide to trust what they see online.

## Getting Started

1. Running on Docker
- Clone the repository
- Download Docker Desktop
- The final model is available for download [here](https://drive.google.com/file/d/1IPYlxYqiQ92cgLi7DAdkQUkfecjd8jX6/view?usp=sharing). Please place the downloaded model in the path `models/deepfake_detection_models/faces_trained_model.h5` to ensure the server runs correctly.
- Run the following command in the root directory of the repository:
```bash
docker build -t deepfake-detection .
docker run -p 5000:5000 deepfake-detection
```
- The service will be accessible at `http://localhost:5000`

2. Running Locally
- Clone the repository
- Install the required dependencies:
```bash
pip install -r requirements.txt
```
- Run the flask server from the root directory of the repository:
```bash
python app.py
```
- The service will be accessible at `http://localhost:5000`

## Contributors

