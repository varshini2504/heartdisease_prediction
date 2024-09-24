import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer  # To handle missing values
import joblib

# Load the dataset
df = pd.read_csv('data/heart.csv')

# Preprocess the data: Convert categorical data into numerical
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df['Smoking'] = df['Smoking'].map({'Current': 1, 'Former': 2, 'Never': 0})
df['Alcohol Intake'] = df['Alcohol Intake'].map({'Heavy': 2, 'Moderate': 1, 'None': 0})
df['Family History'] = df['Family History'].map({'Yes': 1, 'No': 0})
df['Diabetes'] = df['Diabetes'].map({'Yes': 1, 'No': 0})
df['Obesity'] = df['Obesity'].map({'Yes': 1, 'No': 0})
df['Exercise Induced Angina'] = df['Exercise Induced Angina'].map({'Yes': 1, 'No': 0})
df['Chest Pain Type'] = df['Chest Pain Type'].map({'Typical Angina': 1, 'Atypical Angina': 2, 'Non-anginal Pain': 3, 'Asymptomatic': 0})

# Separate features and target
X = df.drop('Heart Disease', axis=1)
y = df['Heart Disease']

# Handle missing values using SimpleImputer
imputer = SimpleImputer(strategy='mean')  # You can use 'median', 'most_frequent', etc.
X = imputer.fit_transform(X)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'heart_disease_model.pkl')

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy * 100:.2f}%')
