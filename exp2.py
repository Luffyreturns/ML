import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix,accuracy_score

data = pd.read_csv(r"C:\Users\omkar\OneDrive\Desktop\ML\dataset\Titanic-Dataset.csv")
print(data.info())
data["Age"] = data["Age"].fillna(data["Age"].mean())

X = data.drop(["Survived","Name","Ticket","Cabin","Embarked"], axis=1)
Y = data['Survived']

X = pd.get_dummies(X, columns=['Sex'], drop_first=True) 

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=10, test_size=0.3)

model = LogisticRegression(max_iter=100)
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

precision = precision_score(Y_test, Y_pred)
recall = recall_score(Y_test, Y_pred)
f1 = f1_score(Y_test, Y_pred)
conf_matrix = confusion_matrix(Y_test, Y_pred)
accuracy = accuracy_score(Y_test, Y_pred)

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(conf_matrix)
