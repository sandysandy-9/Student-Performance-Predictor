import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import resample

# Load dataset
df = pd.read_csv('student_data.csv')

# Display the dataset in Streamlit
st.title("Student Performance Prediction")
st.write("Dataset Preview:")
st.dataframe(df.head())

# Check if the dataset is imbalanced
st.subheader("Class Distribution")
st.write(df['pass_fail'].value_counts())

# Handling Imbalanced Dataset (Optional Step)
# Upsample the minority class
df_majority = df[df.pass_fail == 1]
df_minority = df[df.pass_fail == 0]

df_minority_upsampled = resample(df_minority,
                                 replace=True,    # Sample with replacement
                                 n_samples=len(df_majority), # Match majority class size
                                 random_state=42) # For reproducibility

# Combine the majority class with the upsampled minority class
df_balanced = pd.concat([df_majority, df_minority_upsampled])

# Split the data into features and target
X = df_balanced[['hours_studied', 'attendance', 'past_score', 'extra_activities']]
y = df_balanced['pass_fail']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Model Accuracy
accuracy = accuracy_score(y_test, y_pred)

# Show accuracy and performance metrics
st.subheader("Model Accuracy")
st.write(f"Accuracy: {accuracy * 100:.2f}%")

# Show classification report
st.subheader("Classification Report")
st.text(classification_report(y_test, y_pred))

# User Input Fields for Prediction
st.sidebar.header("Enter Student Information")

# Inputs for the prediction (user enters data)
hours_studied = st.sidebar.slider("Hours Studied", 0, 10, 1)
attendance = st.sidebar.slider("Attendance (%)", 0, 100, 50)
past_score = st.sidebar.slider("Past Score", 0, 100, 50)
extra_activities = st.sidebar.selectbox("Extra Activities (0 = No, 1 = Yes)", [0, 1])

# When user clicks 'Predict', make prediction
if st.sidebar.button('Predict'):
    # Prepare the input data and scale it
    input_data = np.array([[hours_studied, attendance, past_score, extra_activities]])
    scaled_input = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(scaled_input)
    
    # Display the result
    if prediction == 1:
        st.success("The student is predicted to pass!")
    else:
        st.error("The student is predicted to fail.")
