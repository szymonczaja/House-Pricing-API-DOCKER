# 🏠 House Price Prediction Project

## 📝 Project Overview

This project aims to build a production-ready Machine Learning pipeline for predicting house prices using the Ames housing dataset. Unlike typical academic projects, this solution focuses on MLOps best practices, demonstrating key skills in model deployment, custom data processing, and pipeline management.

The core objective is to move beyond simple modeling to create a robust, deployable artifact that handles real-world data challenges through custom transformers and containerization.

---

## 🚀 Key Technologies Used

The solution leverages a modern tech stack to ensure the model is scalable, reproducible, and easy to deploy:

<p align="center">
  <img src="https://img.shields.io/badge/-Python-3776AB?style=flat&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/-FastAPI-009688?style=flat&logo=fastapi&logoColor=white">
  <img src="https://img.shields.io/badge/-Docker-2496ED?style=flat&logo=docker&logoColor=white">
  <img src="https://img.shields.io/badge/-MLflow-0194E2?style=flat&logo=mlflow&logoColor=white">
  <img src="https://img.shields.io/badge/-scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white">
  <img src="https://img.shields.io/badge/-Pandas-150458?style=flat&logo=pandas&logoColor=white">
</p>

* **FastAPI:** Creates a high-performance, robust API service for handling prediction requests with automatic validation.
* **Docker:** Ensures consistency across environments by packaging the application, model, and dependencies into an isolated container.
* **MLflow:** Manages the lifecycle of the ML model, logging the complete pipeline to ensure consistency between training and deployment.
* **Scikit-learn / Custom Transformers:** Drives the core prediction logic with specialized transformers for data cleaning and feature creation.

---

## 🔬 Project Deep Dive: The Data Science Process

The complete analytical process is documented in `notebooks/house_pricing.ipynb`:

1.  **Exploratory Data Analysis (EDA):**
    * Initial deep dive to understand data distributions.
    * Identification of missing values and key correlations driving price.

2.  **Custom Feature Engineering (Critical Step):**
    * Custom Scikit-learn transformers were developed to handle complex preprocessing within the pipeline.
    * **`GroupedMedianTransformer`:** Implements smart imputation based on neighborhood medians rather than global averages.
    * **`FeatureTransformer`:** Generates powerful new features (e.g., Total Square Footage, House Age) to boost model performance.

3.  **Model Prototyping:**
    * Experimentation with various algorithms to identify the optimal regressor for the dataset.

4.  **Model Registration:**
    * The final, optimized model—along with its entire preprocessing pipeline—is logged and saved using **MLflow**, making it ready for immediate serving.

5.  **Model Explainability (SHAP):**
    * Integration of **SHAP (SHapley Additive exPlanations)** to interpret predictions, providing transparency and building trust by explaining *why* a specific price was predicted.

---

## 🛠️ How to Run the API (Using Docker)

To run the API and test the model, you only need Docker Desktop installed.

1. Build the Container Image

In the main project directory, build the image:


docker build -t house-pricing-api:latest .

2. Run the Container

Run the image, mapping the internal container port 8000 to your local machine:


docker run -d --name house-pricing-production -p 8000:8000 house-pricing-api:latest

3. Test the API Endpoints

Once the container is running, the API documentation (Swagger UI) is available at:

👉 http://localhost:8000/docs

Use the POST /predict endpoint with the required house data JSON to get a prediction.
