import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("diabetes_final_dataset.csv")
diabetes_df = df[df["Condition"] == "Diabetes"].copy()

print("Diabetes Dataset:")
print(diabetes_df.head())

print("Dataset Information:")
diabetes_df.info()

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
sns.histplot(diabetes_df["Age"], kde=True)
plt.title("Age Distribution")

plt.subplot(2, 2, 2)
sns.boxplot(x=diabetes_df["Cholesterol"])
plt.title("Cholesterol Distribution")

plt.subplot(2, 2, 3)
sns.scatterplot(x="Age", y="Cholesterol", data=diabetes_df)
plt.title("Age vs Cholesterol")

plt.subplot(2, 2, 4)
features = [
    "Age",
    "Cholesterol",
    "Systolic_BP",
    "Diastolic_BP"
]
sns.heatmap(
    diabetes_df[features].corr(),
    annot=True,
    cmap="viridis"
)
plt.title("Correlation Heatmap")

plt.tight_layout()
plt.show()

X = diabetes_df[
    ["Age", "Cholesterol", "Systolic_BP", "Diastolic_BP"]
]
y = diabetes_df["Gender_Male"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Model Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
