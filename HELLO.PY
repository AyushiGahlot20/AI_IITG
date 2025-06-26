import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Load your CSV files
train_df = pd.read_csv("Train_Data.csv")
test_df = pd.read_csv("Test_Data.csv")

# Drop rows with missing target
train_df = train_df.dropna(subset=['age_group'])

# Encode target: Adult = 0, Senior = 1
train_df['age_group'] = train_df['age_group'].map({'Adult': 0, 'Senior': 1})

# Features used
features = [
    'RIDAGEYR', 'RIAGENDR', 'PAQ605', 'BMXBMI',
    'LBXGLU', 'DIQ010', 'LBXGLT', 'LBXIN'
]

# Prepare train and test sets
X_train = train_df[features]
y_train = train_df['age_group']
X_test = test_df[features]

# Pipeline: Imputer + RandomForest
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the model
pipeline.fit(X_train, y_train)

# Predict on test set
predictions = pipeline.predict(X_test)

# Save to final_submission.csv
submission_df = pd.DataFrame({'age_group': predictions})
submission_df.to_csv("final_submission1.csv", index=False)

print("✅ final_submission.csv created successfully.")
