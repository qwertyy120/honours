import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("Life Expectancy Data.csv")
df.columns = df.columns.str.strip()

df = df.dropna(subset=["Life expectancy"])
df["Outcome"] = np.where(df["Life expectancy"] >= df["Life expectancy"].median(), "High", "Low")

print("Dataset Shape:", df.shape)
print("\nLife Expectancy Outcome Distribution:")
print(df["Outcome"].value_counts())
print(df.head())

features = [
    "Adult Mortality", "infant deaths", "Alcohol", "percentage expenditure",
    "Hepatitis B", "Measles", "BMI", "under-five deaths", "Polio",
    "Total expenditure", "Diphtheria", "HIV/AIDS", "GDP", "Population",
    "thinness  1-19 years", "thinness 5-9 years",
    "Income composition of resources", "Schooling"
]

X = df[features].fillna(df[features].median())
y = (df["Outcome"] == "High").astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Random Forest Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Low", "High"]))

explainer = shap.TreeExplainer(model)
X_sample = X_test.sample(300, random_state=42)
shap_values = explainer.shap_values(X_sample)
sv = shap_values[:, :, 1] if np.ndim(shap_values) == 3 else shap_values[1]

importance = pd.DataFrame({
    "Feature": features,
    "Mean |SHAP|": np.abs(sv).mean(axis=0)
}).sort_values("Mean |SHAP|", ascending=False)

print("Feature Importance (Mean |SHAP value|):")
print(importance.to_string(index=False))

shap.summary_plot(sv, X_sample, plot_type="bar", show=False)
plt.title("SHAP Feature Importance")
plt.tight_layout()
plt.show()

shap.summary_plot(sv, X_sample, show=False)
plt.title("SHAP Summary Plot")
plt.tight_layout()
plt.show()
