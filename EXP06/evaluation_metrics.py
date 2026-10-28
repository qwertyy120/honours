import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

filename = "healthcare_uncleaned_data.csv"
df = pd.read_csv(filename)
print(df.head())
print("\nDataset Shape:", df.shape)

df.columns = df.columns.str.strip()
df = df.drop(
    ['Patient Name', 'Email', 'Phone Number'],
    axis=1,
    errors='ignore'
)

bp = df['Blood Pressure'].str.split('/', expand=True)
df['Systolic_BP'] = pd.to_numeric(bp[0], errors='coerce')
df['Diastolic_BP'] = pd.to_numeric(bp[1], errors='coerce')
df = df.drop('Blood Pressure', axis=1)

df['Visit Date'] = pd.to_datetime(df['Visit Date'], errors='coerce')
df['Visit_Year'] = df['Visit Date'].dt.year
df = df.drop('Visit Date', axis=1)

df['Age'] = df['Age'].replace(0, np.nan)
df['Cholesterol'] = df['Cholesterol'].replace(0, np.nan)
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Cholesterol'] = df['Cholesterol'].fillna(df['Cholesterol'].median())
df['Systolic_BP'] = df['Systolic_BP'].fillna(df['Systolic_BP'].median())
df['Diastolic_BP'] = df['Diastolic_BP'].fillna(df['Diastolic_BP'].median())
df['Visit_Year'] = df['Visit_Year'].fillna(df['Visit_Year'].median())

df['Outcome'] = np.where(df['Cholesterol'] >= 200, 1, 0)

df = pd.get_dummies(
    df,
    columns=['Gender', 'Condition', 'Medication'],
    drop_first=True
)

X = df.drop(['Outcome', 'Cholesterol'], axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model = DecisionTreeClassifier(
    random_state=42,
    max_depth=5
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("Evaluation Metrics")
print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=['Normal', 'High'],
    zero_division=0
))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Normal', 'High'],
    yticklabels=['Normal', 'High']
)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
