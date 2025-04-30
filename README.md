# Student Performance Predictor

## Project Overview
This project predicts the performance of students based on various features such as study hours, past grades, attendance, and extra activities. The goal of this project is to predict whether a student will pass or fail based on their performance data.

The dataset includes:
- `study_hours`: Number of hours the student studies per day.
- `attendance`: Percentage of classes attended by the student.
- `past_score`: Previous score obtained by the student in exams.
- `extra_activities`: Whether the student participated in extracurricular activities (0 = No, 1 = Yes).
- `pass_fail`: The target variable (0 = Fail, 1 = Pass).

The project uses a machine learning model to predict student outcomes, leveraging algorithms like Logistic Regression and Random Forest for classification.

## Model Overview
The model used in this project is trained using a dataset of student performance. It predicts whether a student will pass or fail based on their study hours, attendance, past scores, and participation in extra activities.

### Accuracy and Performance
- **Training Accuracy**: 100% (This might be due to overfitting on the small dataset).
- **Test Accuracy**: ~66.67%
- **Precision, Recall, F1-Score** (Test Data):
    - **Class 0 (Fail)**:
        - Precision: 0.00
        - Recall: 0.00
        - F1-Score: 0.00
    - **Class 1 (Pass)**:
        - Precision: 0.67
        - Recall: 1.00
        - F1-Score: 0.80
    - **Accuracy**: 0.67 (67%)

### Important Files in the Project:
- `student_data.csv`: The dataset used for training the model.
- `app.py`: The main script to run the Streamlit application and interact with the user.
- `notebook.ipynb`: The Jupyter Notebook used for training and testing the model.
- `scaler.pkl`: A pickle file that stores the trained StandardScaler used for feature scaling.
- `student_performance_model.pkl`: The trained machine learning model saved as a pickle file.

## How to Use

### Prerequisites
To run this project, ensure you have the following installed on your system:
- Python 3.x
- pip (Python package manager)

You also need to install the required dependencies for the project. This can be done using the `requirements.txt` file.

### Installation
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/sandysandy-9/Student-Performance-Predictor.git
   cd Student-Performance-Predictor
 2.Create a Virtual Environment (Optional but recommended):
    python -m venv venv
    
3.Activate the Virtual Environment:
On Windows:
bash
Copy code
.\venv\Scripts\activate
On Mac/Linux:
bash
Copy code
source venv/bin/activate
4.Install Required Packages: Install the required libraries by running:
bash
Copy code
pip install -r requirements.txt

5.Run the Application: Run the Streamlit app:
bash
Copy code
streamlit run app.py

Files in this Project
Here are the important files included in this project:
student_data.csv: The dataset used to train the model. It contains the performance data of students.
app.py: The main application code that runs the Streamlit interface for interacting with the user.
notebook.ipynb: Jupyter Notebook that includes the step-by-step process of model training, testing, and evaluation.
retrain_model.py: Script to retrain the model when new data is available.
scaler.pkl: The trained StandardScaler model to scale new input data.
student_performance_model.pkl: The trained machine learning model saved using pickle for prediction.

Model Performance
The performance of the model is evaluated using various metrics, including accuracy, precision, recall, and F1-score. Below is a summary of the model’s performance on the test dataset:

Model Accuracy: 66.67%

Confusion Matrix:

lua
Copy code
[[0 1]
 [0 2]]
Classification Report:

markdown
Copy code
precision    recall  f1-score   support

         0       0.00      0.00      0.00         1
         1       0.67      1.00      0.80         2

  accuracy                           0.67         3

 macro avg       0.33      0.50      0.40         3
weighted avg 0.44 0.67 0.53 3

