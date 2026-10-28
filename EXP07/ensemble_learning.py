import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

filename = "healthcare_uncleaned_data.csv"
df = pd.read_csv(filename)

print("Dataset Shape:", df.shape)
print("\nFirst 5 Records:")
print(df.head())

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

X = df.drop('Condition', axis=1)
y = df['Condition']

print("Disease Classes:")
print(y.value_counts())

numerical_features = [
    'Age',
    'Cholesterol',
    'Systolic_BP',
    'Diastolic_BP',
    'Visit_Year'
]

categorical_features = [
    'Gender',
    'Medication'
]

numeric_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(
            handle_unknown='ignore'
        ))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training Samples:", len(X_train))
print("Testing Samples :", len(X_test))

model1 = LogisticRegression(
    max_iter=1000
)
model2 = DecisionTreeClassifier(
    random_state=42,
    max_depth=5
)
model3 = SVC(
    probability=True,
    random_state=42
)

lr_model = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('classifier', model1)
    ]
)
dt_model = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('classifier', model2)
    ]
)
svm_model = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('classifier', model3)
    ]
)

print("Individual Model Accuracies")
print("-" * 40)

individual_models = [
    (lr_model, 'Logistic Regression'),
    (dt_model, 'Decision Tree'),
    (svm_model, 'SVM')
]

for model, name in individual_models:
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(
        y_test,
        predictions
    )
    print(
        f"Accuracy of {name}: "
        f"{accuracy:.4f}"
    )

ensemble_voting = VotingClassifier(
    estimators=[
        ('lr', lr_model),
        ('dt', dt_model),
        ('svm', svm_model)
    ],
    voting='hard'
)
ensemble_voting.fit(
    X_train,
    y_train
)
voting_predictions = ensemble_voting.predict(
    X_test
)
voting_accuracy = accuracy_score(
    y_test,
    voting_predictions
)
print("-" * 40)
print(
    f"Accuracy of Voting Ensemble: "
    f"{voting_accuracy:.4f}"
)

rf_model = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ))
    ]
)
rf_model.fit(
    X_train,
    y_train
)
rf_predictions = rf_model.predict(
    X_test
)
rf_accuracy = accuracy_score(
    y_test,
    rf_predictions
)
print(
    f"Accuracy of Random Forest Ensemble: "
    f"{rf_accuracy:.4f}"
)

print("Random Forest Classification Report")
print("-" * 40)
print(
    classification_report(
        y_test,
        rf_predictions
    )
)
