
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Step 1: Load the dataset
data = pd.read_csv("students.csv")

print("Dataset loaded successfully!")
print(data.head())

# Step 2: Select input features
features = [
    "attendance",
    "study_hours",
    "previous_marks",
    "assignments_completed"
]

X = data[features]

# Step 3: Select the output column
y = data["result"]

# Step 4: Divide data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# Step 5: Create the Decision Tree model
model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

# Step 6: Train the model
model.fit(X_train, y_train)

print("\nModel training completed!")

# Step 7: Predict test results
y_pred = model.predict(X_test)

# Step 8: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred,
        labels=["Fail", "Pass"]
    )
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

# Step 9: Save the trained model
os.makedirs("model", exist_ok=True)

joblib.dump(
    {
        "model": model,
        "features": features
    },
    "model/student_model.pkl"
)

print("\nTrained model saved successfully!")
print("File: model/student_model.pkl")
