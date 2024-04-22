# Deepfake Detection as a Microservice

## Introduction

In an era where digital content is ubiquitous, the proliferation of deepfake technology poses unprecedented challenges to the integrity of information online. Deepfakes—synthetic media in which a person's likeness has been manipulated to create convincing, yet entirely fictional content—have the potential to mislead, defraud, and undermine public discourse. Recognizing the critical need for robust countermeasures, our project, "Deepfake Detection as a Microservice," aims to provide a powerful, accessible solution to this modern dilemma.

This repository houses a cutting-edge microservice designed to identify and flag deepfake content with precision and speed. Leveraging advanced machine learning and image processing techniques, our service is engineered to integrate seamlessly with various platforms, offering a vital tool for content creators, social media platforms, and news organizations to ensure the authenticity of their media.

## Getting Started

1. Running on Docker
- Clone the repository
- Download Docker Desktop
- The final model is available for download [here](https://drive.google.com/file/d/1IPYlxYqiQ92cgLi7DAdkQUkfecjd8jX6/view?usp=sharing). Please place the downloaded model in the path `models/deepfake_detection_models/faces_trained_model.h5` to ensure the server runs correctly.
- Run the following command in the root directory of the repository:
```bash
docker build -t deepfake-detection .
docker run -p 5000:8080 deepfake-detection
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
- The service will be accessible at `http://localhost:8080`

## Results
Our model was evaluated on the test dataset using two approaches:

| Approach                        | Accuracy | Precision | Recall | True Negative Rate |
|---------------------------------|----------|-----------|--------|--------------------|
| Balanced Fake-to-Real Ratio (1.5:1) | 80.82%   | 80.97%    | 88.93% | 68.66%             |
| Whole Test Dataset              | 83.71%   | 94.03%    | 86.34% | 68.66%             |

### Sample Output
<table>
  <tr>
    <td>
      <img src="https://github.com/suvanbalu/Deepfake-Detection-Microservice/blob/main/test/real1.png" alt="Real Image" width="400"/>
    </td>
    <td>
      <img src="https://github.com/suvanbalu/Deepfake-Detection-Microservice/blob/main/test/fake1.png" alt="Fake Image" width="400"/>
    </td>
  </tr>
  <tr>
    <td>
      <b>Real Image Results:</b>
      <pre>
{
  "face_detection_method": "SSD",
  "predictions": [
    {
      "prediction": "Real",
      "probability": 0.9765076637268066
    }
  ]
}
      </pre>
    </td>
    <td>
      <b>Fake Image Results:</b>
      <pre>
{
  "face_detection_method": "SSD",
  "predictions": [
    {
      "prediction": "Fake",
      "probability": 0.6263623237609863
    }
  ]
}
      </pre>
    </td>
  </tr>
</table>

These results validate the model's effectiveness in accurately detecting deepfake content under varied test conditions.

## Future Works
