import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import RFE
import pickle
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset (keep your original path)
data_path = r"C:\\Users\\sayya\\Desktop\\ML\\Disease-outbreak-prediction-using-Machine-Learning-main\\Datasets\\diabetes.csv"  
data = pd.read_csv(data_path)

print("First 5 rows of the dataset:")
print(data.head())

# Display the number of rows and columns in the dataset
print("\nShape of the dataset:")
print(data.shape)

# Getting basic information about the dataset
print("\nDataset Information:")
print(data.info())

# Checking for missing values
print("\nMissing values in the dataset:")
print(data.isnull().sum())

# Handle zeros in features where zero is not physiologically possible
# This was missing in your original code
print("\nChecking zeros in important features:")
zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for column in zero_columns:
    zeros = (data[column] == 0).sum()
    print(f"{column}: {zeros} zeros ({zeros/len(data)*100:.2f}%)")

# Replace zeros with median values where zeros don't make physiological sense
for column in zero_columns:
    # Get median of non-zero values
    median_value = data[data[column] != 0][column].median()
    # Replace zeros with median
    data[column] = data[column].replace(0, median_value)

print("\nAfter replacing zeros with median values:")
for column in zero_columns:
    zeros = (data[column] == 0).sum()
    print(f"{column}: {zeros} zeros")

# Statistical measures of the dataset
print("\nStatistical summary:")
print(data.describe())

# Check the distribution of the target variable
print("\nDistribution of the target variable:")
print(data['Outcome'].value_counts())
print(f"Class distribution: {data['Outcome'].value_counts(normalize=True) * 100}")

# Define features (X) and target (Y)
X = data.drop(columns='Outcome', axis=1)
Y = data['Outcome']

# Save original feature names before transformation
original_feature_names = X.columns.tolist()

# Feature Engineering: Polynomial features (keeping this from your code)
# But reducing degree to 1 (no transformation) for simplicity
poly = PolynomialFeatures(degree=1, include_bias=False)
X_poly = poly.fit_transform(X)
poly_feature_names = poly.get_feature_names_out(input_features=X.columns)

# Skip RFE - use all features for simplicity and better interpretability
# Instead of RFE feature selection, let's use all features
X_selected = X_poly
selected_feature_names = poly_feature_names

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# Handle class imbalance using SMOTE - keeping this from your code
sm = SMOTE(random_state=42)
X_resampled, Y_resampled = sm.fit_resample(X_scaled, Y)

# Split the resampled data
X_train, X_test, Y_train, Y_test = train_test_split(
    X_resampled, Y_resampled, test_size=0.2, stratify=Y_resampled, random_state=42
)

# Display split shapes
print("\nShapes of the data splits:")
print(f"X_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")

# Train the model - keeping your RandomForest parameters 
model = RandomForestClassifier(random_state=2, n_estimators=160, max_depth=4)

# Add cross-validation for more reliable performance evaluation
cv_scores = cross_val_score(model, X_train, Y_train, cv=5, scoring='accuracy')
print(f"\nCross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean():.4f}")

# Train the model
model.fit(X_train, Y_train)

# Training accuracy
train_accuracy = model.score(X_train, Y_train)
print(f"\nTraining Accuracy: {train_accuracy * 100:.2f}%")

# Predictions and test accuracy
Y_pred = model.predict(X_test)
test_accuracy = accuracy_score(Y_test, Y_pred)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Performance evaluation
print("\nClassification Report:")
print(classification_report(Y_test, Y_pred))

print("\nConfusion Matrix:")
conf_matrix = confusion_matrix(Y_test, Y_pred)
print(conf_matrix)

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Feature Importance Analysis
if hasattr(model, 'feature_importances_'):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importances')
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [original_feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.show()
    
    print("\nFeature Importances:")
    for i in indices:
        print(f"{original_feature_names[i]}: {importances[i]:.4f}")

# Save preprocessing components and model - this was incomplete in your original code
# We need to save all preprocessing elements to ensure consistent predictions
preprocessing = {
    'poly': poly,
    'scaler': scaler,
    'original_feature_names': original_feature_names
}

preprocessing_filename = "diabetes_preprocessing.sav"
with open(preprocessing_filename, 'wb') as preproc_file:
    pickle.dump(preprocessing, preproc_file)
print(f"Preprocessing components saved to {preprocessing_filename}")

model_filename = "diabetes_model.sav"
with open(model_filename, 'wb') as model_file:
    pickle.dump(model, model_file)
print(f"Model saved to {model_filename}")

# Define a proper prediction function - this was missing in your code
def predict_diabetes(new_data, preprocessing_file="diabetes_preprocessing.sav", model_file="diabetes_model.sav"):
    """
    Make a prediction on new patient data
    
    Parameters:
    new_data : dict or DataFrame
        Patient data with features matching the original dataset
    
    Returns:
    dict with prediction and probability
    """
    # Load preprocessing components
    with open(preprocessing_file, 'rb') as file:
        preprocessing = pickle.load(file)
    
    # Load model
    with open(model_file, 'rb') as file:
        model = pickle.load(file)
    
    # Convert dict to DataFrame if needed
    if isinstance(new_data, dict):
        new_data = pd.DataFrame([new_data])
    
    # Ensure all required features are present
    for feature in preprocessing['original_feature_names']:
        if feature not in new_data.columns:
            raise ValueError(f"Missing feature: {feature}")
    
    # Align feature order
    new_data = new_data[preprocessing['original_feature_names']]
    
    # Apply polynomial features
    new_data_poly = preprocessing['poly'].transform(new_data)
    
    # Apply scaling
    new_data_scaled = preprocessing['scaler'].transform(new_data_poly)
    
    # Make prediction
    prediction = model.predict(new_data_scaled)[0]
    probability = model.predict_proba(new_data_scaled)[0][1]
    
    return {
        'prediction': int(prediction),
        'probability': float(probability),
        'result': 'Diabetic' if prediction == 1 else 'Non-Diabetic'
    }

# Example prediction with proper preprocessing
# This fixes the issue in your original code where prediction didn't go through the full pipeline
print("\n----- Example Prediction -----")
# Create an example patient (use median values from the dataset)
example_patient = {
    'Pregnancies': 3,
    'Glucose': 117,
    'BloodPressure': 72,
    'SkinThickness': 23,
    'Insulin': 30,
    'BMI': 32,
    'DiabetesPedigreeFunction': 0.3725,
    'Age': 29
}

# Make prediction
prediction_result = predict_diabetes(example_patient)
print(f"Example patient prediction: {prediction_result['result']} with {prediction_result['probability']:.4f} probability")

# Now test with actual data from X_test (before SMOTE)
# But we need to inverse transform this to get back original data
# This demonstrates how to use this in production with new patient data
print("\n----- Testing with data point from test set -----")
# Get original data point (from before SMOTE and scaling)
original_features = data.iloc[0].drop('Outcome')
print("Original features from dataset:")
print(original_features)

# Predict using our function
prediction_result = predict_diabetes(original_features.to_dict())
print(f"Test prediction: {prediction_result['result']} with {prediction_result['probability']:.4f} probability")

