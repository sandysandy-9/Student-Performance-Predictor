from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
import pandas as pd

# Load your dataset
df = pd.read_csv('student_data.csv')

# Select the features you want to use for prediction
X = df[['hours_studied', 'attendance', 'past_score', 'extra_activities']]  # Features
y = df['pass_fail']  # Target variable

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features (standardize them)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit on training data, then transform
X_test_scaled = scaler.transform(X_test)  # Only transform test data

# Initialize and train the model (Logistic Regression)
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# Test the model on the test data
y_pred = model.predict(X_test_scaled)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy}")

# Save the trained model and scaler
joblib.dump(model, 'student_performance_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
