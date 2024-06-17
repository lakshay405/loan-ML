import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Loading the dataset into a pandas DataFrame
loan_dataset = pd.read_csv('/content/dataset.csv')

# Displaying the first 5 rows of the dataframe
print(loan_dataset.head())

# Displaying the number of rows and columns in the dataset
print(loan_dataset.shape)

# Statistical summary of the dataset
print(loan_dataset.describe())

# Checking for missing values in each column
print(loan_dataset.isnull().sum())

# Dropping rows with missing values
loan_dataset = loan_dataset.dropna()

# Checking for missing values again after dropping
print(loan_dataset.isnull().sum())

# Label encoding for 'Loan_Status'
loan_dataset.replace({"Loan_Status": {'N': 0, 'Y': 1}}, inplace=True)

# Displaying the first 5 rows after encoding
print(loan_dataset.head())

# Checking the count of values in the 'Dependents' column
print(loan_dataset['Dependents'].value_counts())

# Replacing '3+' with '4' in 'Dependents' column
loan_dataset = loan_dataset.replace(to_replace='3+', value=4)

# Checking the count of values again after replacement
print(loan_dataset['Dependents'].value_counts())

# Visualizing Education vs Loan_Status
sns.countplot(x='Education', hue='Loan_Status', data=loan_dataset)

# Visualizing Marital Status vs Loan_Status
sns.countplot(x='Married', hue='Loan_Status', data=loan_dataset)

# Converting categorical columns to numerical values
loan_dataset.replace({
    'Married': {'No': 0, 'Yes': 1},
    'Gender': {'Male': 1, 'Female': 0},
    'Self_Employed': {'No': 0, 'Yes': 1},
    'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2},
    'Education': {'Graduate': 1, 'Not Graduate': 0}
}, inplace=True)

# Separating the data and labels
X = loan_dataset.drop(columns=['Loan_ID', 'Loan_Status'], axis=1)
Y = loan_dataset['Loan_Status']

print(X)
print(Y)

# Splitting the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

# Initializing and training the Support Vector Machine (SVM) classifier
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

# Accuracy score on training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy on training data:', training_data_accuracy)

# Accuracy score on test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy on test data:', test_data_accuracy)

# Predicting for new data
new_data = {
    'Gender': 1,
    'Married': 1,
    'Dependents': 0,
    'Education': 1,
    'Self_Employed': 0,
    'ApplicantIncome': 5000,
    'CoapplicantIncome': 2000,
    'LoanAmount': 150,
    'Loan_Amount_Term': 360,
    'Credit_History': 1,
    'Property_Area': 2
}

# Converting to DataFrame
new_data_df = pd.DataFrame(new_data, index=[0])

# Making predictions using the trained classifier
new_prediction = classifier.predict(new_data_df)
print(f'Prediction for new data: {"Approved" if new_prediction[0] == 1 else "Not Approved"}')

# Predicting for user input data
new_data = {
    'Gender': int(input("Enter Gender (1 for Male, 0 for Female): ")),
    'Married': int(input("Enter Marital Status (1 for Yes, 0 for No): ")),
    'Dependents': int(input("Enter Number of Dependents: ")),
    'Education': int(input("Enter Education (1 for Graduate, 0 for Not Graduate): ")),
    'Self_Employed': int(input("Enter Self Employed status (1 for Yes, 0 for No): ")),
    'ApplicantIncome': float(input("Enter Applicant Income: ")),
    'CoapplicantIncome': float(input("Enter Coapplicant Income: ")),
    'LoanAmount': float(input("Enter Loan Amount: ")),
    'Loan_Amount_Term': float(input("Enter Loan Amount Term: ")),
    'Credit_History': float(input("Enter Credit History (1 for Yes, 0 for No): ")),
    'Property_Area': int(input("Enter Property Area (2 for Urban, 1 for Semiurban, 0 for Rural): "))
}

# Converting to DataFrame
new_data_df = pd.DataFrame(new_data, index=[0])

# Making predictions using the trained classifier
new_prediction = classifier.predict(new_data_df)
print(f'Prediction for new data: {"Approved" if new_prediction[0] == 1 else "Not Approved"}')
