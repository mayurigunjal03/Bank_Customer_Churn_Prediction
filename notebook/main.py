import pandas as pd
import matplotlib.pyplot as plt
# Load Dataset
df = pd.read_csv("data/European_Bank.csv")
print(df.head())
print(df.columns)
# Remove unwanted columns
df.drop(["CustomerId", "Surname"], axis=1, inplace=True)
plt.figure(figsize=(6,4))
df["Exited"].value_counts().plot(kind="bar")
plt.title("Churn Distribution")
plt.xlabel("Exited")
plt.ylabel("Count")
plt.show()
plt.figure(figsize=(8,5))
df.groupby("Age")["Exited"].mean().plot()
plt.title("Age vs Churn Rate")
plt.xlabel("Age")
plt.ylabel("Average Churn Rate")
plt.show()
plt.figure(figsize=(8,5))
df.groupby("Geography")["Exited"].mean().plot(kind="bar")
plt.title("Germany vs Churn Rate")
plt.xlabel("Germany")
plt.ylabel("Average Churn Rate")
plt.show()
plt.figure(figsize=(8,5))
df.groupby("Gender")["Exited"].mean().plot(kind="bar")
plt.title("Gender vs Churn Rate")
plt.xlabel("Gender")
plt.ylabel("Average Churn Rate")
plt.show()
# Feature Engineering
df["BalanceSalaryRatio"] = df["Balance"] / (df["EstimatedSalary"] + 1)
df["ProductDensity"] = df["NumOfProducts"] / (df["Age"] + 1)
df["EngagementProduct"] = df["IsActiveMember"] * df["NumOfProducts"]
df["AgeTenure"] = df["Age"] * df["Tenure"]
# Convert text columns into numbers
df = pd.get_dummies(df, drop_first=True)
print(df.head())
print(df.columns)
# Features and Target
X = df.drop("Exited", axis=1)
y = df["Exited"]
print(X.shape)
print(y.shape)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print(X_train.shape)
print(X_test.shape)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(X_train.shape)
print(X_test.shape)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
# Logistic Regression
lr = LogisticRegression(
    max_iter=1000,
    random_state=42
)
lr.fit(X_train, y_train)
print("Logistic Regression Training Completed")
# Decision Tree

dt = DecisionTreeClassifier(
    random_state=42
)

dt.fit(X_train, y_train)

print("Decision Tree Training Completed")

# Random Forest

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("Random Forest Training Completed")
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

# Logistic Regression Evaluation

lr_pred = lr.predict(X_test)

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

lr_accuracy = accuracy_score(y_test, lr_pred)

print("Logistic Accuracy =", lr_accuracy)

lr_prob = lr.predict_proba(X_test)[:,1]

lr_auc = roc_auc_score(y_test, lr_prob)

print("Logistic ROC-AUC =", lr_auc) 
# Decision Tree Evaluation

dt_pred = dt.predict(X_test)

dt_accuracy = accuracy_score(y_test, dt_pred)

print("Decision Tree Accuracy =", dt_accuracy)

dt_prob = dt.predict_proba(X_test)[:,1]

dt_auc = roc_auc_score(y_test, dt_prob)

print("Decision Tree ROC-AUC =", dt_auc)
import pandas as pd
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})
feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)
feature_importance.to_csv("feature_importance.csv", index=False)
print("Feature Importance Saved")
# Prediction
y_pred = model.predict(X_test)
print(y_pred[:10])
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy =", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
y_prob = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_prob)
print("ROC-AUC Score =", roc_auc)
import pickle
# Save Model
pickle.dump(model, open("model.pkl", "wb"))
# Save Scaler
pickle.dump(scaler, open("scaler.pkl", "wb"))
print("Model Saved Successfully")
print("Scaler Saved Successfully")