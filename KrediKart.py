import pandas as pd

df = pd.read_csv('creditcard.csv')

X = df.iloc[:, :-1]

y = df.iloc[:, -1]

sinif_dagilimi = y.value_counts()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

log_model = LogisticRegression(random_state=42)
log_model.fit(X_train_scaled, y_train)

tree_model = DecisionTreeClassifier(random_state=42, max_depth=5)
tree_model.fit(X_train_scaled, y_train)

from sklearn.metrics import confusion_matrix, classification_report

y_pred_log = log_model.predict(X_test_scaled)
print("Lojistik Regresyon Confusion Matrix:\n", confusion_matrix(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))

y_pred_tree = tree_model.predict(X_test_scaled)
print("Karar Ağaci Confusion Matrix:\n", confusion_matrix(y_test, y_pred_tree))
print(classification_report(y_test, y_pred_tree))





