🤖 ATS Score Predictor: AI Resume–Job Description Matching

🌟 Project Overview

This project implements a machine learning solution to predict the compatibility score between a candidate's resume and a specific job description (JD), mimicking the functionality of an Applicant Tracking System (ATS).

The goal is to provide a fast, lightweight, and deployable model that can instantly score the relevance of application materials. The entire pipeline, from feature engineering to deployment, is optimized for CPU efficiency.

Key Goals Achieved

High Accuracy: Achieved an MAE of approximately 15.63, significantly improving the baseline.

Low Compute: Utilizes the highly efficient LightGBM model combined with sparse TF-IDF features.

Full Deployment: Packaged into a Docker container for portable, real-time inference via a Flask API.

📝 Problem Statement (ATS Compatibility)

Recruiters rely heavily on Applicant Tracking Systems (ATS) to screen hundreds of applications by checking for keyword matches, relevant skills, and contextual alignment between the resume and the job description.

The core challenge is to model this complex textual relationship and predict a single continuous ATS compatibility score (the target variable) given the combined text of the resume and the JD.

Output: A compatibility score (e.g., 0 to 100), where a higher score indicates a stronger fit.

🔗 Project Notebook & Analysis

The full data cleaning, exploratory data analysis (EDA), model selection, and hyperparameter search is documented in the accompanying Colab notebook.

Notebook Link: [[colab notebook]](https://colab.research.google.com/drive/1ukfhg_ZHctoWxfPyX_TtN7hfWB93nYgE?usp=sharing)

📦 Dataset

Source: Resume-ATS Score v1 dataset (Hugging Face)
Size: Approximately 6,000 Resume–JD pairs.
Format: Each sample contains the combined text input and the target numerical score.
| Field | Description |
| :--- | :--- |
| text | The combined, pre-cleaned string of the Resume and Job Description. |
| ats_score | The target variable (continuous numerical score, 0-100). |

Data Accessibility:
dataset = load_dataset("0xnbk/resume-ats-score-v1-en")
dataset

🔍 Machine Learning Approach

1. Feature Engineering

We use TF-IDF (Term Frequency-Inverse Document Frequency) for feature extraction, which naturally emphasizes keywords and unique terminology critical for ATS-like matching.

2. Model Selection & Evaluation

We compared several models on the validation set.

Model

RMSE

MAE

Notes

Baseline (TF-IDF LR)

23.33

18.67

Initial simple linear model.

Ridge Regression

21.93

18.46

Marginal stability improvement.

LightGBM (Final Model)

19.72

15.63

Selected for superior performance and speed.

The LightGBM model achieved the best accuracy, resulting in a 16.3% reduction in MAE over the baseline, making it the optimal choice for this deployment.

🚀 Deployment and API Instructions

The final model is served using a Flask API and containerized with Docker, ensuring easy deployment and reproducibility.

Required Files

The repository should contain the following executable and asset files:

File Name

Purpose

Requirement Fulfilled

train.py

Script to train and save the final LightGBM model assets.

Training Script

predict.py

Flask API script to handle real-time /predict requests.

Prediction Script (Web Service)

requirements.txt

List of all Python dependencies (flask, lightgbm, joblib, etc.).

Dependencies

Dockerfile

Instructions to build the containerized service.

Containerization

How to Run Locally (Using Docker)

Build the Docker Image:

docker build -t ats-score-predictor .


Run the Container:

docker run -it --rm -p 9699:9699 ats-score-predictor


The service will start on port 9699.

Test the API Endpoint:
Use curl to send a sample text input (combined resume and job description):

curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"text": "Candidate has 5 years experience in Python and NLP. JD requires 3+ years in Python, Flask, and ML models."}' \
  http://localhost:9699/predict


The API will respond with the predicted ats_score.
