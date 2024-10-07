import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load data
train_data = pd.read_csv('C:/Temp/Titanic/train.csv')
test_data = pd.read_csv('C:/Temp/Titanic/test.csv')

# Combine train and test data for preprocessing
combined_data = pd.concat([train_data, test_data], axis=0, ignore_index=True)

# Feature Engineering
# Example: Extract titles from names
combined_data['Title'] = combined_data['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)

# Example: Group ages into categories
combined_data['Age_Category'] = pd.cut(combined_data['Age'], bins=[0, 18, 40, 60, 100], labels=['Child', 'Adult', 'Middle-Aged', 'Elderly'])

# Example: Encode categorical variables
label_encoder = LabelEncoder()
for col in ['Sex', 'Embarked', 'Age_Category', 'Title']:
    combined_data[col] = label_encoder.fit_transform(combined_data[col].astype(str))

# Split combined data back into train and test sets
train_data = combined_data.iloc[:len(train_data)]
test_data = combined_data.iloc[len(train_data):].reset_index(drop=True)

# Prepare data for modeling
X = train_data.drop(['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
y = train_data['Survived']
X_test = test_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived'], axis=1)

# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Selection and Hyperparameter Tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Evaluate best model
best_rf = grid_search.best_estimator_
val_predictions = best_rf.predict(X_val)
val_accuracy = accuracy_score(y_val, val_predictions)
print("Validation Accuracy:", val_accuracy)

# Make predictions on test data
test_predictions = best_rf.predict(X_test)

# Save submission file
submission = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': test_predictions})
submission.to_csv('C:/temp/submission_higher_score.csv', index=False)
print("Submission file saved successfully.")
