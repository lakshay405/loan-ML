# loan-ML
Loan Approval Prediction Using Support Vector Machine (SVM)
This project aims to predict loan approval status using machine learning techniques, specifically Support Vector Machine (SVM) classifier.

Dataset
The dataset (dataset.csv) contains information about loan applicants and whether their loan was approved (Loan_Status):

Features include attributes like gender, marital status, dependents, education, income details, loan amount, credit history, and property area.
The target variable (Loan_Status) indicates whether the loan was approved (1) or not (0).
Workflow
Data Loading and Exploration:

Load the loan dataset from a CSV file into a Pandas DataFrame (loan_dataset).
Display dataset summary including the first few rows, dimensions, statistical summary, and check for missing values.
Clean the dataset by dropping rows with missing values and encode categorical variables (Loan_Status, Married, Gender, Self_Employed, Property_Area, Education).
Data Visualization:

Visualize relationships between loan approval (Loan_Status) and categorical variables (Education, Married).
Data Preprocessing:

Separate the dataset into features (X) and the target variable (Y).
Split the data into training and testing sets using train_test_split.
Model Training and Evaluation:

Initialize an SVM classifier (SVC with linear kernel) and train it using the training data (X_train, Y_train).
Evaluate the model's accuracy on both training and test sets using accuracy_score.
Prediction:

Demonstrate the model's predictive capabilities by providing example input data (new_data).
Output the prediction result based on whether the loan application is predicted to be approved or not.
Libraries Used
numpy, pandas for data manipulation and analysis.
seaborn for data visualization.
sklearn for model selection (SVC), evaluation (train_test_split, accuracy_score), and preprocessing.
Conclusion
This project demonstrates the application of SVM for predicting loan approval based on applicant information. By training the model on historical loan data and evaluating its accuracy, it provides a tool for financial institutions to assess the likelihood of approving a loan application, contributing to efficient decision-making processes.

