import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

df = pd.read_csv("gameplay_data.csv")

X = df.drop(columns=["action"])
y = df["action"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

labels = ["up", "down", "left", "right"]

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nConfusion matrix (rows=actual, cols=predicted):")
print(f"{'':>8}", "  ".join(f"{l:>5}" for l in labels))
cm = confusion_matrix(y_test, y_pred, labels=labels)
for label, row in zip(labels, cm):
    print(f"{label:>8}", "  ".join(f"{v:>5}" for v in row))

joblib.dump(model, "model.pkl")
print("\nModel saved to model.pkl")
