
import pandas as pd
import numpy as np
import matplotlib.pyplot as pl

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
df = pd.read_csv(r"G:\Titanic-Dataset.csv")

print("==== TITANIC DATASET ====")
print(df.shape)
print(df.columns)
print(df.describe())
print(df.info())
print(df.isnull().sum())

# Fill missing values
print("MODE")
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Age'].fillna(df['Age'].median(), inplace=True)

# Drop Cabin column
if 'Cabin' in df.columns:
    df.drop('Cabin', axis=1, inplace=True)

# Encode categorical columns
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

le = LabelEncoder()
df["Embarked"] = le.fit_transform(df["Embarked"])

# Drop unnecessary columns
df.drop(["PassengerId", "Name", "Ticket"], axis=1, inplace=True)

print(df.head())

# Survival Count
pl.figure(figsize=(5, 4))
df['Survived'].value_counts().plot(kind='bar')
pl.title("Survival Count")
pl.xlabel("Survived")
pl.ylabel("Count")
pl.show()

# Age Histogram
print("HISTOGRAM")
pl.figure(figsize=(6, 4))
pl.hist(df['Age'], bins=30)
pl.title("Age Distribution")
pl.xlabel("Age")
pl.ylabel("Frequency")
pl.show()

# Fare Histogram
pl.figure(figsize=(6, 4))
pl.hist(df['Fare'], bins=30)
pl.title("Fare Distribution")
pl.xlabel("Fare")
pl.ylabel("Frequency")
pl.show()

pl.figure(figsize=(8, 6))
corr = df.corr()

pl.imshow(corr, cmap='coolwarm')
pl.colorbar()

pl.xticks(range(len(corr.columns)), corr.columns, rotation=90)
pl.yticks(range(len(corr.columns)), corr.columns)

pl.title("Correlation Matrix")
pl.show()

X = df.drop("Survived", axis=1)
y = df["Survived"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(X_scaled[:5])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.20,
    random_state=79
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

#Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

#Prediction
y_pred = model.predict(X_test)

#Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

new_passenger=[[
    3,
    0,
    25,
    0, 
    0,
    8.05,
    2
]]
new_scaled=scaler.transform(new_passenger)
prediction=model.predict(new_scaled)

if prediction[0]==1:
    print("Passsenger survived")
else:
    print("Passenger not survived")
