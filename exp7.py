import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = pd.read_csv(r"C:\Users\omkar\OneDrive\Desktop\ML\dataset\adult_dataset.csv")

data.replace('?', pd.NA, inplace=True)

label_encoders = {}
categorical_cols = ['workclass', 'education', 'marital.status', 'occupation', 
                    'relationship', 'race', 'sex', 'native.country', 'income']

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))  
    label_encoders[col] = le

numerical_cols = ['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']

for col in numerical_cols:
    data[col] = data[col].fillna(data[col].mean())

X = data.drop('income', axis=1)  # All columns except 'income'
y = data['income']  # Target variable (income)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Logistic Regression without PCA
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)

# Calculate and display accuracy without PCA
accuracy_without_pca = accuracy_score(y_test, y_pred)
print("Logistic Regression without PCA")
print(f"Accuracy: {accuracy_without_pca:.4f}")
print(classification_report(y_test, y_pred))

pca_all = PCA()
X_train_pca_all = pca_all.fit_transform(X_train)
X_test_pca_all = pca_all.transform(X_test)

log_reg_pca_all = LogisticRegression()
log_reg_pca_all.fit(X_train_pca_all, y_train)
y_pred_pca_all = log_reg_pca_all.predict(X_test_pca_all)

# Calculate and display accuracy with all components PCA
accuracy_pca_all = accuracy_score(y_test, y_pred_pca_all)
print("\nLogistic Regression with PCA (whole dataset)")
print(f"Accuracy: {accuracy_pca_all:.4f}")
print(classification_report(y_test, y_pred_pca_all))

# PCA with 50% explained variance
pca_50 = PCA(0.5)  # Retain components that explain 50% of the variance
X_train_pca_50 = pca_50.fit_transform(X_train)
X_test_pca_50 = pca_50.transform(X_test)

log_reg_pca_50 = LogisticRegression()
log_reg_pca_50.fit(X_train_pca_50, y_train)
y_pred_pca_50 = log_reg_pca_50.predict(X_test_pca_50)

# Calculate and display accuracy with 50% explained variance
accuracy_pca_50 = accuracy_score(y_test, y_pred_pca_50)
print("\nLogistic Regression with PCA (variance explained ≥ 0.5)")
print(f"Accuracy: {accuracy_pca_50:.4f}")
print(classification_report(y_test, y_pred_pca_50))

# PCA with 75% explained variance
pca_75 = PCA(0.75)  # Retain components that explain 75% of the variance
X_train_pca_75 = pca_75.fit_transform(X_train)
X_test_pca_75 = pca_75.transform(X_test)

log_reg_pca_75 = LogisticRegression()
log_reg_pca_75.fit(X_train_pca_75, y_train)
y_pred_pca_75 = log_reg_pca_75.predict(X_test_pca_75)

# Calculate and display accuracy with 75% explained variance
accuracy_pca_75 = accuracy_score(y_test, y_pred_pca_75)
print("\nLogistic Regression with PCA (variance explained ≥ 0.75)")
print(f"Accuracy: {accuracy_pca_75:.4f}")
print(classification_report(y_test, y_pred_pca_75))