Advanced AI Medical Intelligence Platform

AI-assisted chest X-ray classification for Normal vs Pneumonia, with ResNet18, Grad-CAM explainability, FastAPI, Streamlit, Groq-powered report generation, SQLite prediction history, and Docker Compose.

Research / educational prototype only. This system is not a medical device and is not intended for diagnosis, treatment, or clinical decision-making. Model probabilities are not measures of clinical certainty, and Grad-CAM is an attribution visualization rather than a clinically validated lesion map.

Overview

This project takes a chest X-ray image and runs it through an end-to-end AI application:

Upload a chest X-ray.

Preprocess the image for the trained ResNet18 classifier.

Predict Normal or Pneumonia.

Return class probabilities.

Generate a Grad-CAM visualization showing regions that influenced the model output.

Generate an AI-assisted text report using Groq.

Store the prediction and report in SQLite for history/detail retrieval.

Architecture

                         Browser
                            |
                            v
                  +--------------------+
                  | Streamlit Frontend |
                  |       :8501        |
                  +---------+----------+
                            |
                    HTTP / JSON / file
                            |
                            v
                  +--------------------+
                  |    FastAPI API     |
                  |       :8000        |
                  +----+----+----+-----+
                       |    |    |
                       |    |    +----------------+
                       |    |                     |
                       v    v                     v
                  ResNet18  Grad-CAM          Groq LLM
                       |    |                     |
                       +----+----------+----------+
                                    |
                                    v
                               SQLite DB
                                    |
                                    v
                           Prediction History

Docker architecture

The application is split into two Docker Compose services:

medical-api — FastAPI, model inference, Grad-CAM, Groq report generation, SQLite access.

medical-frontend — Streamlit UI.

The frontend communicates with the API container through the Docker Compose service name http://api:8000.

Machine Learning Model

Architecture: ResNet18

Task: Binary chest X-ray classification

Classes:

0 — Normal

1 — Pneumonia

Input size: 224 × 224

Training device: NVIDIA GeForce GTX 1650 (local development/training)

Production container: CPU PyTorch runtime

Training dataset split

Split

Samples

Training

4,708

Validation

524

Test

624

Class weighting

The training run used class weights to compensate for class imbalance:

Class

Weight

Normal

1.9390

Pneumonia

0.6737

Model Results

Validation

Best validation F1: 0.9936

Best epoch: 6

Final test set

Metric

Score

Accuracy

0.9311

Precision

0.9026

Sensitivity / Recall

0.9974

Specificity

0.8205

F1 Score

0.9476

ROC-AUC

0.9912

Test confusion matrix

                Predicted
               Normal Pneumonia
Actual Normal    192      42
       Pneumonia  1      389

The model is highly sensitive to pneumonia in this test run, but specificity is lower than sensitivity. These metrics should be interpreted as model evaluation results, not clinical performance guarantees.

Threshold Analysis

A validation threshold sweep was performed instead of assuming the default 0.50 threshold.

The selected validation threshold was 0.30, with validation F1 of approximately 0.9936.

Selected validation results:

Threshold

Accuracy

Precision

Recall

F1

Specificity

0.25

0.9866

0.9872

0.9949

0.9910

0.9630

0.30

0.9905

0.9923

0.9949

0.9936

0.9778

0.50

0.9905

0.9948

0.9923

0.9936

0.9852

0.85

0.9885

0.9974

0.9871

0.9922

0.9926

Explainability — Grad-CAM

Grad-CAM is used to visualize which image regions contributed to the model's prediction.

The implementation:

hooks into the target ResNet18 convolutional layer,

computes gradients for the selected class,

produces a normalized heatmap,

overlays the heatmap on the original X-ray,

exposes the resulting image through the API/UI.

Important limitation: Grad-CAM explains model attribution. It does not prove that a highlighted region is a pneumonia lesion or disease location.

LLM Report Generation

The application uses the Groq API to turn verified classifier output into a human-readable AI-assisted report.

The report includes:

AI model prediction

Model probability

Explainability availability

Interpretation of model output

Medical disclaimer

The prompt is deliberately constrained so that the LLM does not independently diagnose the patient or invent symptoms, history, laboratory values, imaging findings, treatment recommendations, or other unsupported medical information.

Secret handling

Local development uses .env.

Docker uses a Compose secret at /run/secrets/groq_api_key.

.env and .secrets/ must never be committed to Git.

API Endpoints

Health check

GET /health

Example response:

{
  "status": "healthy"
}

Prediction

POST /predict

Consumes multipart/form-data with an X-ray image.

Example response fields include:

{
  "id": 11,
  "prediction": "Pneumonia",
  "class_id": 1,
  "probability": 0.9989,
  "normal_probability": 0.0011,
  "pneumonia_probability": 0.9989,
  "gradcam_available": true,
  "gradcam_url": "/files/gradcam_api/...png",
  "report": "...",
  "model_version": "resnet18-v1",
  "created_at": "..."
}

Prediction history

GET /history

Returns previously stored predictions.

Prediction detail

GET /prediction/{prediction_id}

Returns the stored result for a specific prediction ID.

Database

SQLite is used for application persistence because this project primarily needs structured records for prediction history and detail lookup.

The database stores information such as:

prediction ID

filename

predicted class

class probabilities

Grad-CAM path

generated report

model version

creation timestamp

A vector database is not required for this application's current inference/history workflow because the application is not performing semantic document retrieval. A relational database is a better fit for these structured records.

Project Structure

advanced-ai-medical-platform/
│
├── app/
│   ├── api/
│   │   ├── routes_prediction.py
│   │   └── routes_history.py
│   ├── database/
│   │   ├── crud.py
│   │   ├── database.py
│   │   └── models.py
│   ├── llm/
│   │   └── report_generator.py
│   ├── ml/
│   │   ├── dataset.py
│   │   ├── gradcam.py
│   │   ├── gradcam_visualization.py
│   │   ├── model.py
│   │   ├── prediction_service.py
│   │   └── preprocessing.py
│   ├── schemas/
│   │   └── prediction.py
│   └── main.py
│
├── frontend/
│   └── app.py
│
├── model/
│   └── best_resnet18.pth
│
├── results/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── gradcam_api/
│
├── scripts/
│   ├── analyze_dataset.py
│   ├── analyze_threshold.py
│   ├── evaluate_model.py
│   ├── generate_gradcam.py
│   ├── init_database.py
│   ├── inspect_dataset.py
│   ├── train.py
│   └── visualize_dataset.py
│
├── tests/
│   ├── test_database.py
│   ├── test_database_insert.py
│   ├── test_dataloader.py
│   ├── test_gradcam.py
│   ├── test_llm.py
│   ├── test_model.py
│   └── test_prediction_service.py
│
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .gitignore
├── .env.example
├── requirements.txt
└── requirements-gpu.txt

Exact files can change as the project evolves; the core runtime structure above describes the implemented application architecture.

Local Development

1. Activate the virtual environment

.\.venv\Scripts\Activate.ps1

2. Install dependencies

python -m pip install -r requirements.txt

For local GPU training, use the GPU-specific environment/dependency setup used by the training workflow.

3. Configure environment variables

Create .env from .env.example and set your Groq configuration:

GROQ_API_KEY=your_key_here
GROQ_MODEL=llama-3.3-70b-versatile

Never commit .env.

4. Initialize the database

python -m scripts.init_database

5. Start the API

python -m uvicorn app.main:app --reload

API:

http://127.0.0.1:8000

Swagger UI:

http://127.0.0.1:8000/docs

6. Start the Streamlit frontend

streamlit run frontend/app.py

Frontend:

http://localhost:8501

Docker

Docker Compose starts both services:

docker compose up

Or detached mode:

docker compose up -d

Check service status:

docker compose ps

Stop services:

docker compose down

Expected local URLs:

FastAPI: http://localhost:8000

Swagger: http://localhost:8000/docs

Streamlit: http://localhost:8501

The project has been tested locally with Docker Compose using separate API and frontend containers.

Testing

The project includes focused test scripts for:

model loading

data loading

Grad-CAM generation

prediction service

database initialization and insertion

Groq LLM report generation

Examples:

python -m tests.test_model
python -m tests.test_gradcam
python -m tests.test_prediction_service
python -m tests.test_database
python -m tests.test_database_insert
python -m tests.test_llm

Generated Artifacts

Important generated outputs include:

results/confusion_matrix.png
results/roc_curve.png
results/gradcam_api/

Deployment Notes

The application has been containerized successfully with Docker Compose and verified locally with both services running.

For a public deployment, the API and frontend should be treated as separate services. Render supports Docker-based web services and recommends separate services for separate applications rather than trying to run multiple apps in a single service. The backend should bind to 0.0.0.0 and a Render-assigned port. See the official Render documentation before deployment.

Limitations

Binary classification is limited to Normal vs Pneumonia.

Performance metrics come from this project's test split and are not evidence of clinical effectiveness.

Model probabilities are not calibrated clinical probabilities unless separately validated and calibrated.

Grad-CAM is an attribution technique, not a validated lesion segmentation/localization method.

The LLM report is generated from verified model output and must not be treated as a medical diagnosis.

The current application is a research/educational prototype.

License

Add the license required by the assignment or repository owner before publishing publicly.