
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# URL of the raw CSV file
url = 'titanic_csv.csv'

# Load the dataset
try:
    data = pd.read_csv(url)
    print("Data loaded successfully.")
except Exception as e:
    print(f"Error loading data: {e}")

# Display the first few rows of the dataset
print(data.head())

# Define features and target variable
# For the Titanic dataset, we'll predict survival
X = data.drop(columns=['Survived', 'Name', 'Ticket', 'Cabin', 'PassengerId'])
y = data['Survived']

# Define categorical and numerical columns
categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(include=['number']).columns

# Create preprocessing pipelines for numerical and categorical data
numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ]
)

# Preprocess the data
X_processed = preprocessor.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Define and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred))

# Plot feature importances
importances = model.feature_importances_
features = preprocessor.get_feature_names_out()
indices = np.argsort(importances)

plt.figure(figsize=(10, 6))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Feature Importance')
plt.show()

# Function to predict survival based on user input
def predict_survival(model, preprocessor):
    print("\nEnter passenger details to predict survival:")
    pclass = int(input("Pclass (1, 2, or 3): "))
    sex = input("Sex (male or female): ")
    age = float(input("Age: "))
    sibsp = int(input("Number of siblings/spouses aboard: "))
    parch = int(input("Number of parents/children aboard: "))
    fare = float(input("Fare: "))
    embarked = input("Port of Embarkation (C, Q, or S): ")

    # Create a DataFrame with the input data
    input_data = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [sex],
        'Age': [age],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [fare],
        'Embarked': [embarked]
    })

    # Preprocess the input data
    input_processed = preprocessor.transform(input_data)

    # Predict survival
    prediction = model.predict(input_processed)

    # Output the result
    if prediction[0] == 1:
        print("The passenger would have survived.")
    else:
        print("The passenger would not have survived.")

# Interactive menu
def main():
    while True:
        print("\nMenu:")
        print("1. View model performance")
        print("2. Predict survival for a passenger")
        print("3. Exit")

        choice = input("Enter your choice (1, 2, or 3): ")

        if choice == '1':
            # Display model performance
            print(f"Accuracy: {accuracy * 100:.2f}%")
            print(classification_report(y_test, y_pred))
        elif choice == '2':
            # Predict survival
            predict_survival(model, preprocessor)
        elif choice == '3':
            print("Exiting...")
            break
        else:
            print("Invalid choice, please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()