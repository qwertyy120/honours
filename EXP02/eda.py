import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("diabetes_final_dataset.csv")

diabetes_df = df[df["Condition"] == "Diabetes"]

print("Diabetes Dataset Head:")
print(diabetes_df.head())

print("Dataset Info:")
diabetes_df.info()

print("Descriptive Statistics:")
print(diabetes_df.describe())

print("\nMissing Values:")
print(diabetes_df.isnull().sum())

print("\nDuplicate Records:")
print(diabetes_df.duplicated().sum())

import matplotlib.pyplot as plt
import pandas as pd

diabetes_df = df[df["Condition"] == "Diabetes"]
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.hist(diabetes_df["Age"], bins=8)
plt.xlabel("Age")
plt.ylabel("Count")
plt.title("Age Distribution of Diabetic Patients")

plt.subplot(2, 2, 2)
gender_count = [
    diabetes_df["Gender_Male"].sum(),
    diabetes_df["Gender_Female"].sum()
]
plt.bar(
    ["Male", "Female"],
    gender_count
)
plt.xlabel("Gender")
plt.ylabel("Count")
plt.title("Gender Count of Diabetic Patients")

plt.subplot(2, 2, 3)
plt.hist(diabetes_df["Cholesterol"], bins=8)
plt.xlabel("Cholesterol")
plt.ylabel("Count")
plt.title("Cholesterol Distribution")

plt.subplot(2, 2, 4)
plt.boxplot([
    diabetes_df["Systolic_BP"],
    diabetes_df["Diastolic_BP"]
])
plt.xticks(
    [1, 2],
    ["Systolic BP", "Diastolic BP"]
)
plt.ylabel("Blood Pressure")
plt.title("Blood Pressure Distribution")
plt.tight_layout()
plt.show()
