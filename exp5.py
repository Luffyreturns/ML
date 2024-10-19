import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import lightgbm as lgb

data = pd.read_csv(r"C:\Users\omkar\OneDrive\Desktop\ML\dataset\adult_dataset.csv")

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

# Step 7: Train and Evaluate AdaBoost model
ada_model = AdaBoostClassifier(n_estimators=50, random_state=42)
ada_model.fit(X_train, y_train)
ada_pred = ada_model.predict(X_test)
print("AdaBoost Results:")
print(f"Accuracy: {accuracy_score(y_test, ada_pred)}")
print(classification_report(y_test, ada_pred))

# Step 8: Train and Evaluate Gradient Boosting model
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)
print("\nGradient Boosting Results:")
print(f"Accuracy: {accuracy_score(y_test, gb_pred)}")
print(classification_report(y_test, gb_pred))

# Step 9: Train and Evaluate XGBoost model
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
print("\nXGBoost Results:")
print(f"Accuracy: {accuracy_score(y_test, xgb_pred)}")
print(classification_report(y_test, xgb_pred))

# Step 10: Train and Evaluate LightGBM model
lgb_model = lgb.LGBMClassifier(n_estimators=100, random_state=42)
lgb_model.fit(X_train, y_train)
lgb_pred = lgb_model.predict(X_test)
print("\nLightGBM Results:")
print(f"Accuracy: {accuracy_score(y_test, lgb_pred)}")
print(classification_report(y_test, lgb_pred))
