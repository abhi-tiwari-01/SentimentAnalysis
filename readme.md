# Emotion Detection from Punjabi-English Mixed Social Media Posts

This project involves building a model to detect emotions from social media posts that are written in a mix of English and Punjabi. It utilizes machine learning and natural language processing techniques to classify the sentiment as either **negative**, **neutral**, or **positive**.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Model Performance](#model-performance)
- [Confusion Matrix](#confusion-matrix)
- [Publication Trends](#publication-trends)
- [Technology Stack](#technology-stack)
- [Setup](#setup)
- [Usage](#usage)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project aims to build a sentiment analysis model specifically for text written in a mix of English and Punjabi, focusing on social media posts. The model was trained on a dataset with mixed-language posts and tested using various classification techniques.

## Features
- Sentiment classification of Punjabi-English mixed social media posts.
- Three sentiment categories: **Negative**, **Neutral**, and **Positive**.
- User-friendly web interface to analyze text.

## Model Performance

The model achieved the following performance metrics:

- **Test Accuracy:** 92%
- **Precision, Recall, F1-Score:**
  - **Negative:** Precision = 0.92, Recall = 0.98, F1-score = 0.95
  - **Neutral:** Precision = 0.97, Recall = 0.35, F1-score = 0.52
  - **Positive:** Precision = 0.92, Recall = 0.86, F1-score = 0.89
  - **Overall Accuracy:** 0.92

Below is the classification report and confusion matrix:

### Classification Report:
| Class    | Precision | Recall | F1-score | Support |
|----------|-----------|--------|----------|---------|
| Negative | 0.92      | 0.98   | 0.95     | 7119    |
| Neutral  | 0.97      | 0.35   | 0.52     | 316     |
| Positive | 0.92      | 0.86   | 0.89     | 2871    |

## Confusion Matrix

The confusion matrix represents the model's performance in identifying the three different sentiment classes:

- **0:** Negative
- **1:** Neutral
- **2:** Positive

![Confusion Matrix](/img/confusion%20matrix.png)

## Publication Trends

This project was developed based on an extensive review of related literature. The figure below shows the percentage distribution of papers by publication source in our domain:

![Distribution of Papers](/img/papers%20chart.png)

Additionally, we observed an increasing trend in publications on this topic over recent years:

![Yearly Publications](/img/year%20vs%20publication.png)

## Technology Stack
- **Language:** Python
- **Machine Learning:** Scikit-learn, TensorFlow
- **Web Framework:** Flask
- **NLP Libraries:** NLTK, SpaCy
- **Visualization:** Matplotlib, Seaborn

## Setup

To run this project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/abhi-tiwari-01/SentimentAnalysis.git
2. Navigate to the project directory:
    ```bash
    cd emotion-detection-punjabi-english
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
4. Run the app:
    ```bash
    python app.py

## Usage

To use the application, simply input a mixed Punjabi-English text in the user interface, and the model will classify the sentiment. Below is a screenshot of the user interface:

![User Interface](/img/User%20Interface.png)

## Screenshots

- **Classification Results:**

    ![Classification Results](/img/classification%20matrix.png)

- **Confusion Matrix:**

    ![Confusion Matrix](/img/confusion%20matrix.png)

- **Publication Trends:**

    ![Publication Trends](/img/year%20vs%20publication.png)

## Contributing

Feel free to fork this repository and contribute by submitting a pull request.

## License

This project is licensed under the MIT License.
