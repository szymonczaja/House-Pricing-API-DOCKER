🏠 House Price Prediction Project

This project aims to build a production-ready Machine Learning pipeline for predicting house prices using the Ames housing dataset. It demonstrates key skills in model deployment, custom data processing, and MLOps tools.

🚀 Key Technologies Used

This solution uses a modern tech stack to ensure the model can be easily managed and deployed:

FastAPI: Used to create a fast and robust API service for handling prediction requests.

Docker: Used for containerization, packaging the application and all dependencies (including the ML model) into an isolated environment for easy deployment.

MLflow: Used for model management. It saves and loads the complete prediction pipeline, ensuring consistency between training and deployment.

Scikit-learn / Custom Transformers: Used to build the core prediction logic.

🔬 Project Deep Dive: The Data Science Process

The notebooks/house_pricing.ipynb file contains the full analytical process:

-Exploratory Data Analysis (EDA): Initial analysis to understand the data, identify missing values, and find key correlations.

-Model Prototyping: Experimentation with various ML algorithms to find the best performing model for price prediction.

-Custom Feature Engineering: Critical Step. We developed custom Scikit-learn transformers (like GroupedMedianTransformer and FeatureTransformer) to handle complex data cleaning (imputation based on neighborhood median) and create powerful new features (e.g., total square footage, house age).

-Model Registration: The final, optimized model and its entire preprocessing pipeline are logged and saved using MLflow.

-Model Explainability (SHAP): We used SHAP (SHapley Additive exPlanations) to interpret the model's predictions, ensuring transparency and trust in the final results.

🛠️ How to Run the API (Using Docker)
To run the API and test the model, you only need Docker Desktop installed.

1. Build the Container Image
In the main project directory, build the image:

docker build -t house-pricing-api:latest .

2. Run the Container
Start the container and map the internal port 8000 to your computer's port 8000:

docker run -d --name house-pricing-production -p 8000:8000 house-pricing-api:latest

3. Test the Prediction Endpoint
Open your browser and navigate to the API documentation:

👉 http://localhost:8000/docs

Use the POST /predict endpoint with the required house data to receive an instant price prediction.

🔭 Future Improvements (Roadmap)

To continue development, the next steps include:

Testing (Pytest): Implement comprehensive unit and integration tests for the custom transformers and the API endpoints to ensure code quality.

LLM Integration: Integrate a Large Language Model (LLM) to generate natural language reports that explain the prediction to the user (e.g., "This house is priced high because of its quality and location").
