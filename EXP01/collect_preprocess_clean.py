import numpy as np
import pandas as pd

df = pd.read_csv("healthcare_uncleaned_data.csv")
print("Dataset loaded successfully!")
print("Initial Shape:", df.shape)

print("First 5 Records:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Records:")
print(df.duplicated().sum())

text_cols = [
    "Patient Name", "Gender",
    "Condition", "Medication", "Email"
]

for col in text_cols:
    df[col] = df[col].astype(str).str.strip().str.title()

df["Age"] = df["Age"].replace(0, np.nan)
df["Cholesterol"] = df["Cholesterol"].replace(0, np.nan)

df["Age"] = df["Age"].fillna(df["Age"].median())
df["Cholesterol"] = df["Cholesterol"].fillna(
    df["Cholesterol"].median()
)

df["Gender"] = df["Gender"].replace({
    "Other": "Unknown",
    "None": "Unknown",
    "Nan": "Unknown"
})

df["Visit Date"] = pd.to_datetime(
    df["Visit Date"],
    errors="coerce"
)

df = df.dropna(subset=["Visit Date"])
df = df.drop_duplicates()

print("After Cleaning:")
print("Shape:", df.shape)
print(df.head())

disease_cohort = df[
    df["Condition"] == "Diabetes"
].copy()

print("Diabetes Dataset:")
print("Number of records:", len(disease_cohort))
print(disease_cohort.head())

patient_data = disease_cohort[
    ["Patient Name", "Age", "Gender", "Email"]
].copy()

clinical_data = disease_cohort[
    ["Condition", "Medication", "Visit Date",
     "Blood Pressure", "Cholesterol"]
].copy()

patient_data["Patient_ID"] = range(
    1, len(patient_data) + 1
)

clinical_data["Patient_ID"] = range(
    1, len(clinical_data) + 1
)

integrated_data = pd.merge(
    patient_data,
    clinical_data,
    on="Patient_ID"
)

print("Integrated Data:")
print(integrated_data.head())

bp = integrated_data[
    "Blood Pressure"
].str.split("/", expand=True)

integrated_data["Systolic_BP"] = pd.to_numeric(bp[0])
integrated_data["Diastolic_BP"] = pd.to_numeric(bp[1])

integrated_data["Age_Group"] = pd.cut(
    integrated_data["Age"],
    bins=[0, 18, 35, 50, 65, 100],
    labels=[
        "Minor",
        "Young Adult",
        "Middle Aged",
        "Senior",
        "Geriatric"
    ]
)

integrated_data["Visit_Year"] = (
    integrated_data["Visit Date"].dt.year
)

integrated_data["Visit_Month"] = (
    integrated_data["Visit Date"].dt.month
)

integrated_data = pd.get_dummies(
    integrated_data,
    columns=["Gender", "Medication", "Age_Group"],
    dtype=int
)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
cols = [
    "Age",
    "Cholesterol",
    "Systolic_BP",
    "Diastolic_BP"
]

integrated_data[cols] = scaler.fit_transform(
    integrated_data[cols]
)

print("Final ML-Ready Dataset:")
print(integrated_data.head())
print("Final Shape:", integrated_data.shape)

integrated_data.to_csv(
    "diabetes_final_dataset.csv",
    index=False
)

print("Final dataset saved successfully!")
