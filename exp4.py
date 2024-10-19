import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # Random Forest instead of Decision Tree
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Step 1: Load the dataset
data = pd.read_csv(r"C:\Users\omkar\OneDrive\Desktop\ML\dataset\adult_dataset.csv")

# Step 2: Data Preprocessing
# Replace '?' with NaN (pd.NA)
data.replace('?', pd.NA, inplace=True)

# Step 3: Convert categorical columns to numerical using LabelEncoder
label_encoders = {}
categorical_cols = ['workclass', 'education', 'marital.status', 'occupation', 
                    'relationship', 'race', 'sex', 'native.country', 'income']

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))  # Ensure all values are strings
    label_encoders[col] = le

# Step 4: Replace NaN with the mean for numerical columns
numerical_cols = ['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']

for col in numerical_cols:
    data[col] = data[col].fillna(data[col].mean())

# Step 5: Define features (X) and target (y)
X = data.drop('income', axis=1)  # All columns except 'income'
y = data['income']  # Target variable (income)

# Step 6: Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train the Random Forest model
model = RandomForestClassifier(
    n_estimators=200,       # number of trees in the forest
    criterion='gini',       # or 'entropy'
    max_depth=10,           # limit the maximum depth of each tree
    min_samples_split=5,    # minimum samples to split a node
    min_samples_leaf=2,     # minimum samples at a leaf node
    max_features='sqrt',    # number of features to consider for the best split
    bootstrap=True,         # use bootstrap samples
    random_state=42,        # for reproducibility
    n_jobs=-1               # use all available CPU cores
)
model.fit(X_train, y_train)

# Step 8: Make predictions
y_pred = model.predict(X_test)

# Step 9: Evaluate the model
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))
