# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
# from sklearn import svm
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# from sklearn.preprocessing import StandardScaler
# import pickle

# data_path = r"C:\Users\sayya\Desktop\ML\Disease-outbreak-prediction-using-Machine-Learning-main\Datasets\parkinsons.csv"  
# data = pd.read_csv(data_path)

# print("First 5 rows of the dataset:")
# data.head()

# # Checking for missing values
# print("\nMissing values in the dataset:")
# print(data.isnull().sum())

# # Define features (X) and target (Y)
# X = data.drop(columns=['name', 'status'], axis=1)  # Dropping 'name' and 'status' columns
# Y = data['status']

# # Standardize the feature data
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# # Split the data into training and testing sets
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# # Display the shapes of the splits
# print("\nShapes of the data splits:")
# print(f"X: {X.shape}")
# print(f"X_train: {X_train.shape}")
# print(f"X_test: {X_test.shape}")

# model = svm.SVC(random_state=2)
# model.fit(X_train, Y_train)

# # Calculate training accuracy
# train_accuracy = model.score(X_train, Y_train)
# print(f"\nTraining Accuracy: {train_accuracy * 100:.2f}%")

# # Make predictions on the test set
# Y_pred = model.predict(X_test)

# # Calculate test accuracy
# test_accuracy = accuracy_score(Y_test, Y_pred)
# print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# # Classification report
# print("\nClassification Report:")
# print(classification_report(Y_test, Y_pred))

# # Confusion matrix
# print("\nConfusion Matrix:")
# print(confusion_matrix(Y_test, Y_pred))

# # Build a predictive system
# sample_data = np.array([X_test[0]])  
# sample_prediction = model.predict(sample_data)
# print("\nSample Prediction:", "Parkinson's" if sample_prediction[0] == 1 else "No Parkinson's")

# # Save the model to a .sav file
# model_filename = "parkinsons_model.sav"
# with open(model_filename, 'wb') as model_file:
#     pickle.dump(model, model_file)

# print(f"\nModel saved to {model_filename}")

# # Save the scaler for future use in prediction
# scaler_filename = "scaler_parkinsons.sav"
# with open(scaler_filename, 'wb') as scaler_file:
#     pickle.dump(scaler, scaler_file)

# print(f"Scaler saved to {scaler_filename}")

# # Print column names
# print("\nColumns in the dataset:")
# data.columns.tolist()

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pickle

# Load the dataset
data_path = r"C:\Users\sayya\Desktop\ML\Disease-outbreak-prediction-using-Machine-Learning-main\Datasets\parkinsons.csv"
data = pd.read_csv(data_path)

print("="*50)
print("📊 First 5 Rows of the Dataset")
print("="*50)
print(data.head())

# Check for missing values
print("\n" + "="*50)
print("🧪 Missing Values in the Dataset")
print("="*50)
print(data.isnull().sum())

# Define features and target
X = data.drop(columns=['name', 'status'], axis=1)
Y = data['status']

# Standardize the feature data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

print("\n" + "="*50)
print("📐 Shapes of the Data Splits")
print("="*50)
print(f"➡️ Features (X): {X.shape}")
print(f"📘 Training Set (X_train): {X_train.shape}")
print(f"📗 Test Set (X_test): {X_test.shape}")

# Model training
model = svm.SVC(random_state=2)
model.fit(X_train, Y_train)

# Training accuracy
train_accuracy = model.score(X_train, Y_train)
print("\n" + "="*50)
print("✅ Training Accuracy")
print("="*50)
print(f"📈 {train_accuracy * 100:.2f}%")

# Test predictions
Y_pred = model.predict(X_test)
test_accuracy = accuracy_score(Y_test, Y_pred)

print("\n" + "="*50)
print("📊 Test Accuracy")
print("="*50)
print(f"📉 {test_accuracy * 100:.2f}%")

# Classification report
print("\n" + "="*50)
print("📋 Classification Report")
print("="*50)
print(classification_report(Y_test, Y_pred))

# Confusion matrix
print("\n" + "="*50)
print("🧩 Confusion Matrix")
print("="*50)
print(confusion_matrix(Y_test, Y_pred))

# Predict on a sample data point
sample_data = np.array([X_test[0]])
sample_prediction = model.predict(sample_data)
print("\n" + "="*50)
print("🔍 Sample Prediction")
print("="*50)
print("Prediction:", "🧠 Parkinson's" if sample_prediction[0] == 1 else "✅ No Parkinson's")

# Save the trained model
model_filename = "parkinsons_model.sav"
with open(model_filename, 'wb') as model_file:
    pickle.dump(model, model_file)
print("\n💾 Model saved to:", model_filename)

# Save the scaler
scaler_filename = "scaler_parkinsons.sav"
with open(scaler_filename, 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)
print("💾 Scaler saved to:", scaler_filename)

# Show dataset columns
print("\n" + "="*50)
print("🧾 Columns in the Dataset")
print("="*50)
print(data.columns.tolist())
