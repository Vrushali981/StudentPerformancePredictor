import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import matplotlib.pyplot as plt

# ════════════════════════════════════════
# STEP 1 — Load Data
# ════════════════════════════════════════
df = pd.read_csv("students.csv")
print("Data loaded! Shape:", df.shape)

# ════════════════════════════════════════
# STEP 2 — Separate Input & Output
# ════════════════════════════════════════
# X = what we give the model (inputs)
X = df[["attendance", "marks1", "marks2", "assignments"]]

# y = what we want to predict (output)
y = df["risk"]

print("\nInputs (X):")
print(X.head(3))
print("\nOutputs (y):")
print(y.head(3))

# ════════════════════════════════════════
# STEP 3 — Split Data into Train & Test
# ════════════════════════════════════════
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,    # 20% for testing
    random_state=42   # same split every run
)

print(f"\nTraining data: {len(X_train)} students")
print(f"Testing data:  {len(X_test)} students")

# Think of it like:
# Train = textbook examples student studies
# Test  = actual exam questions (never seen before!)

# ════════════════════════════════════════
# STEP 4 — Create & Train the Model
# ════════════════════════════════════════
print("\nTraining model... 🤖")

model = RandomForestClassifier(
    n_estimators=100,  # use 100 decision trees
    random_state=42
)

# THIS ONE LINE IS WHERE MACHINE LEARNING HAPPENS!
model.fit(X_train, y_train)

print("Model trained successfully! ✅")

# ════════════════════════════════════════
# STEP 5 — Test the Model
# ════════════════════════════════════════
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f"\nAccuracy: {accuracy * 100:.2f}%")
# Good accuracy = above 85%

# Detailed report
print("\n=== DETAILED REPORT ===")
print(classification_report(y_test, predictions))

# ════════════════════════════════════════
# STEP 6 — See Feature Importance
# ════════════════════════════════════════
# Which factor matters most for prediction?
importance = pd.Series(
    model.feature_importances_,
    index=["attendance","marks1","marks2","assignments"]
).sort_values(ascending=False)

print("\n=== WHICH FACTOR MATTERS MOST? ===")
print(importance)

# Plot it
importance.plot(kind="bar", color="gold")
plt.title("Feature Importance")
plt.ylabel("Importance Score")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

# ════════════════════════════════════════
# STEP 7 — Test on a Real Student
# ════════════════════════════════════════
print("\n=== PREDICT A NEW STUDENT ===")

new_student = pd.DataFrame([[45, 35, 38, 50]], 
    columns=["attendance","marks1","marks2","assignments"])

result = model.predict(new_student)[0]
proba  = model.predict_proba(new_student)[0]
confidence = round(max(proba) * 100, 1)

print(f"Attendance:  45%")
print(f"Marks1:      35/100")
print(f"Marks2:      38/100")
print(f"Assignments: 50/100")
print(f"\nPrediction:  {result} RISK")
print(f"Confidence:  {confidence}%")

# ════════════════════════════════════════
# STEP 8 — Save the Model
# ════════════════════════════════════════
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel saved as model.pkl ✅")
print("Now you can use it in Flask!")