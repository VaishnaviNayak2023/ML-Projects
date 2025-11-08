Iris Flower Species Classifier This project provides a comprehensive, end-to-end Machine Learning solution for classifying iris flowers. It is structured to showcase a complete MLOps-lite pipeline, integrating model training, API deployment, and frontend integration.

Project Objectives Classification: Accurately classify iris flowers into one of three species (Setosa, Versicolor, or Virginica).

**Workflow Demonstration: **Illustrate best practices for separating machine learning logic (backend) from user interaction (frontend).

Technology Stack: Integrate standard professional tools including Scikit-learn, Flask, and Streamlit.

Project Structure The repository is organized to maintain clear separation of concerns for clarity and scalability.

The key files and directories are:

iris-classifier/ ├── src/ │ ├── init.py # Python package marker. │ ├── model_trainer.py # Script for data ingestion, training, and model serialization. │ ├── prediction_api.py # Flask application to serve real-time predictions. │ └── web_app.py # Streamlit application for the user interface. ├── models/ # Directory to store model artifacts. │ ├── iris_knn_model.pkl # Trained model object (K-Nearest Neighbors). │ └── iris_target_names.pkl # Mapping of numerical labels to species names. ├── README.md # Project documentation. ├── requirements.txt # List of Python dependencies. └── run_project.sh # Optional BASH script for automated setup and execution.

Technology Stack Summary The project utilizes the following core components:

Scikit-learn is used for the Machine Learning (training the model).

Joblib is used for Model Persistence (saving the trained model).

Flask serves as the Backend API (creating the prediction endpoint).

**Streamlit **builds the Frontend UI (providing the interactive interface).

Getting Started 1. Prerequisites Ensure you have Python 3.x and the package manager pip installed.

2. Setup and Installation Execute the following commands from the root iris-classifier directory:

Install dependencies:

Bash

pip install -r requirements.txt Create the necessary models directory:

Bash

mkdir models

3. Execution (Simultaneous Operation) The application requires three sequential steps, with the API and UI running concurrently. Use three separate terminal windows or a split terminal feature for execution.

Step A: Model Training This step trains the K-Nearest Neighbors model on the embedded Iris dataset and saves the necessary files.

In Terminal 1:

Bash

python src/model_trainer.py

Step B: Start the Prediction API (Backend) The Flask server is started to host the model and listen for prediction requests on Port 5000.

In Terminal 2:

Bash

python src/prediction_api.py Keep this terminal session active to keep the server running.

Step C: Launch the Web UI (Frontend) The Streamlit interface is launched. It acts as the client that sends data to the running Flask API.

In Terminal 3:

Bash

streamlit run src/web_app.py The application will launch in your default web browser, allowing you to use the interface to interact with the model in real-time.

Model Details Dataset The model is trained on the classic Iris Dataset, which includes 150 samples from the three Iris species.

Algorithm The K-Nearest Neighbors (KNN) algorithm is employed for classification, configured with K=3. This algorithm is simple yet highly effective for this particular dataset, classifying a new flower based on the majority species among its three closest data points.

Input Features The model uses the following four features for classification:

Sepal Length (cm)

Sepal Width (cm)

Petal Length (cm)

Petal Width (cm)

Cleanup To properly shut down the application:

Close the web browser window running the Streamlit application (Terminal 3).

Return to Terminal 2 (running the Flask API) and press Ctrl + C to terminate the server process.
