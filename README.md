# AI Mental Health Support: Depression Risk Detection

This project aims to develop a machine learning model to predict the likelihood of depression based on student survey data. Early detection can aid in timely intervention for mental health challenges.

## Project Overview

The model is trained on the "Student Mental Health" dataset from Kaggle. It uses features such as gender, age, year of study, CGPA, anxiety status, panic attack history, and whether they sought specialist help to predict if a student reports having depression.

**Note:** This model is for educational and illustrative purposes only and should NOT be used for actual medical diagnosis. Always consult with a qualified healthcare professional for any mental health concerns.

## Directory Structure
mental-health-ai/
├── .gitignore
├── README.md
├── data/
│ └── raw/ # Raw, immutable data
│ └── processed/ # Cleaned or transformed data (optional to store)
├── notebooks/ # Jupyter notebooks for exploration, preprocessing, and model training
│ ├── 1_Data_Exploration_Preprocessing.ipynb
│ └── 2_Model_Training_Evaluation.ipynb
├── src/ # Source code for the project
│ ├── init.py
│ ├── data_preprocessing.py (Example module for preprocessing functions)
│ ├── model_training.py (Example module for training functions)
│ └── predict.py (Script to load model and make predictions)
├── models/ # Trained machine learning models and related files
│ ├── random_forest_depression_model.pkl
│ └── model_features.json
├── requirements.txt # Project dependencies
└── main.py (Optional CLI entry point)


## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Kshiti19/mental-health-ai.git
    cd mental-health-ai
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    # Windows
    .\.venv\Scripts\activate
    # macOS/Linux
    source .venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Data Exploration & Training:**
    *   Open the Jupyter notebooks in the `notebooks/` directory (e.g., using VS Code or JupyterLab) to see the data analysis, preprocessing, and model training steps.
    *   `1_Data_Exploration_Preprocessing.ipynb`
    *   `2_Model_Training_Evaluation.ipynb` (This notebook also saves the trained model)

2.  **Making Predictions:**
    *   The `src/predict.py` script demonstrates how to load the saved model and make predictions on new sample data.
    *   Run from the project root directory:
        ```bash
        python src/predict.py
        ```

## Model Details

*   **Model Used:** Random Forest Classifier
*   **Target Variable:** 'Do you have Depression?' (Binary: Yes/No, mapped to 1/0)
*   **Key Features (Example):** Gender, Age, Year of Study, CGPA, Anxiety, Panic Attacks, Sought Specialist. (See `models/model_features.json` for the full list used by the trained model).
*   **Evaluation (on test set with small data):**
    *   Accuracy: [Insert your test accuracy here, e.g., ~70-80% depending on run and data split]
    *   (Add Precision, Recall, F1 for the 'Depression' class if you have them)

## Future Work / Advanced Options

*   **Larger Dataset:** Use a more extensive and diverse dataset for better generalization.
*   **Hyperparameter Tuning:** Optimize the Random Forest model (e.g., using GridSearchCV or RandomizedSearchCV).
*   **Try Other Models:** Experiment with SVM, Gradient Boosting, or simple Neural Networks.
*   **Text Data & Sentiment Analysis:** Incorporate questionnaire free-text responses or data like Reddit posts. This would involve NLP techniques (TF-IDF, Word Embeddings) and potentially sentiment analysis models.
*   **Deep Learning:** For text data or more complex patterns if a very large dataset is available.
*   **Web Interface:** Build a simple web application (e.g., using Flask or Streamlit) for users to input their data and get a risk assessment.
*   **Address Imbalance More Robustly:** Techniques like SMOTE if class imbalance is severe.
*   **Cross-validation:** Use k-fold cross-validation during training for more robust evaluation.

## Disclaimer

This project is for academic and demonstration purposes only. The predictions made by this model are not a substitute for professional medical advice, diagnosis, or treatment. If you have concerns about your mental health, please consult a qualified healthcare provider.


