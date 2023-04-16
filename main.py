#%%
# General Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
import ipywidgets as widgets
from IPython.display import display
from sklearn.model_selection import RandomizedSearchCV



#%%
# Read Spreadsheet and process
df = pd.read_csv("healthcare-dataset-stroke-data.csv")

# Fill missing values in the 'bmi' column with its mean
df['bmi'].fillna(df['bmi'].mean(), inplace=True)

# Convert categorical variables into numerical variables
df['gender'] = df['gender'].replace({'Male':0,'Female':1,'Other':-1}).astype(np.uint8)
df['Residence_type'] = df['Residence_type'].replace({'Rural':0,'Urban':1}).astype(np.uint8)
df['work_type'] = df['work_type'].replace({'Private':0,'Self-employed':1,'Govt_job':2,'children':-1,'Never_worked':-2}).astype(np.uint8)


# Convert categorical variables into numerical variables using pd.get_dummies()
df = pd.get_dummies(df, columns=['ever_married', 'smoking_status'])

# Split the data into features (X) and the target variable (y)
X = df.drop(['id', 'stroke'], axis=1)
y = df['stroke']

# Oversample the minority class (has stroke) using SMOTE
# Apply SMOTE to balance the dataset
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X, y)

#%%

# RandomizedSearchCV is typically faster than GridSearchCV because it does not 
# exhaustively search through all possible combinations of hyperparameters. 
# Instead, it randomly samples a fixed number of combinations from the 
# specified hyperparameter space.


# Google colab is on the slower side, 
# so we use a smaller number of iterations to get similar results and save time.
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Define the hyperparameter grid
param_grid = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False],
    'criterion': ['gini', 'entropy']
}

# Create a base Random Forest model
base_rf = RandomForestClassifier(random_state=42)

# Create the Randomized Search object
random_search = RandomizedSearchCV(estimator=base_rf, param_distributions=param_grid, n_iter=20, cv=5, n_jobs=-1, verbose=1, random_state=42)

# Fit the Randomized Search object to the training data
random_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = random_search.best_params_
print("Best hyperparameters found by Randomized Search:")
print(best_params)

# Train a new Random Forest classifier using the best hyperparameters
best_rf = RandomForestClassifier(**best_params, random_state=42)
best_rf.fit(X_train, y_train)

# Evaluate the model with the best hyperparameters
y_pred = best_rf.predict(X_test)
print("Accuracy (best hyperparameters):", accuracy_score(y_test, y_pred))

print("Classification Report (best hyperparameters):", classification_report(y_test, y_pred))

# Calculate feature importances
importances = best_rf.feature_importances_
feature_importances = pd.DataFrame({'feature': X.columns, 'importance': importances})
feature_importances = feature_importances.sort_values('importance', ascending=False)
print("Feature Importances:")
print(feature_importances)


#%%
# Feauture Importance plot
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importances)
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# Confusion Matrix plot
from sklearn.metrics import confusion_matrix

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=['No Stroke', 'Stroke'], yticklabels=['No Stroke', 'Stroke'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# Correlation Matrix plot
# Calculate the correlation matrix
corr_matrix = df.corr()

# Plot the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

#%%

def process_user_input(patient):
    # Perform data preprocessing on user input
    patient_processed = pd.DataFrame([patient])

    # Encode categorical variables
    patient_processed['gender'] = patient_processed['gender'].replace({'Male': 0, 'Female': 1, 'Other': -1}).astype(np.uint8)
    patient_processed['Residence_type'] = patient_processed['Residence_type'].replace({'Rural': 0, 'Urban': 1}).astype(np.uint8)
    patient_processed['work_type'] = patient_processed['work_type'].replace(
        {'Private': 0, 'Self-employed': 1, 'Govt_job': 2, 'children': -1, 'Never_worked': -2}).astype(np.uint8)

    patient_processed = pd.get_dummies(patient_processed, columns=['ever_married', 'smoking_status'])

    # Add missing columns (if any) and set their values to 0
    missing_cols = set(X.columns) - set(patient_processed.columns)
    for col in missing_cols:
        patient_processed[col] = 0

    # Reorder columns to match the original dataset
    patient_processed = patient_processed[X.columns]

    return patient_processed


# Create input widgets for each feature
gender_widget = widgets.Dropdown(options=['Male', 'Female', 'Other'], description='Gender:')
age_widget = widgets.FloatText(description='Age:')
hypertension_widget = widgets.Dropdown(options=[('No', 0), ('Yes', 1)], description='Hypertension:')
heart_disease_widget = widgets.Dropdown(options=[('No', 0), ('Yes', 1)], description='Heart Disease:')
ever_married_widget = widgets.Dropdown(options=['Yes', 'No'], description='Ever Married:')
work_type_widget = widgets.Dropdown(options=['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'], description='Work Type:')
residence_type_widget = widgets.Dropdown(options=['Rural', 'Urban'], description='Residence Type:')
avg_glucose_level_widget = widgets.FloatText(description='Avg Glucose Level:')
bmi_widget = widgets.FloatText(description='BMI:')
smoking_status_widget = widgets.Dropdown(options=['never smoked', 'formerly smoked', 'smokes'], description='Smoking Status:')

# Display the widgets
widgets_to_display = [gender_widget, age_widget, hypertension_widget, heart_disease_widget, ever_married_widget,
                      work_type_widget, residence_type_widget, avg_glucose_level_widget, bmi_widget, smoking_status_widget]
for widget in widgets_to_display:
    display(widget)

# Create a button to submit the user input
submit_button = widgets.Button(description="Submit")
display(submit_button)

# Define the button click event
def on_submit_button_click(button):
    # Get user input from widgets
    patient = {
        'gender': gender_widget.value,
        'age': age_widget.value,
        'hypertension': hypertension_widget.value,
        'heart_disease': heart_disease_widget.value,
        'ever_married': ever_married_widget.value,
        'work_type': work_type_widget.value,
        'Residence_type': residence_type_widget.value,
        'avg_glucose_level': avg_glucose_level_widget.value,
        'bmi': bmi_widget.value,
        'smoking_status': smoking_status_widget.value
    }

    # Process user input
    processed_patient = process_user_input(patient)

    # Predict the probability of stroke using the trained model
    stroke_probability = best_rf.predict_proba(processed_patient)[:, 1][0]

    # Display the result
    print("\nThe likelihood of stroke for the entered patient is: {:.2%}".format(stroke_probability))

# Set the button click event
submit_button.on_click(on_submit_button_click)
# %%
