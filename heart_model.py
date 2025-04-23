import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load dataset (keep your original path)
data_path = r"C:\\Users\\sayya\\Desktop\\ML\\Disease-outbreak-prediction-using-Machine-Learning-main\\Datasets\\heart.csv"  
data = pd.read_csv(data_path)

# Split features and target
X = data.drop("target", axis=1)
y = data["target"]

# Identify categorical and numerical columns
categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
numerical_cols = [col for col in X.columns if col not in categorical_cols]

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_cols),
    ("cat", OneHotEncoder(drop='first'), categorical_cols)
])

# Full pipeline with a regularized Random Forest
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=120,
        max_depth=7,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    ))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Fit the pipeline
pipeline.fit(X_train, y_train)

# Accuracy
train_accuracy = pipeline.score(X_train, y_train)
test_accuracy = pipeline.score(X_test, y_test)
print(f"Training Accuracy: {train_accuracy*100:.4f}%")
print(f"Testing Accuracy: {test_accuracy*100:.4f}%")

# Predict and confusion matrix
y_pred = pipeline.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Disease", "Disease"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

# 1. Target distribution
plt.figure(figsize=(6, 4))
sns.countplot(x="target", data=data, palette="viridis")
plt.title("Target Distribution")
plt.xlabel("Heart Disease (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# 2. Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# 3. Age vs. Max Heart Rate (thalach)
plt.figure(figsize=(8, 6))
sns.scatterplot(data=data, x="age", y="thalach", hue="target", palette="Set1")
plt.title("Age vs Max Heart Rate by Heart Disease")
plt.tight_layout()
plt.show()