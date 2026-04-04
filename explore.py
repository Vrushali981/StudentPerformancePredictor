import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("students.csv")

# ─── STEP 1: Basic info ─────────────────────────
print("=== DATASET INFO ===")
print(df.shape)        # rows and columns
print(df.dtypes)       # data types
print(df.describe())   # min, max, mean of each column

# ─── STEP 2: Check missing values ───────────────
print("\n=== MISSING VALUES ===")
print(df.isnull().sum())
# Should show 0 for all — no missing data!

# ─── STEP 3: Count each risk level ──────────────
print("\n=== RISK DISTRIBUTION ===")
print(df["risk"].value_counts())

# ─── STEP 4: Plot risk distribution ─────────────
plt.figure(figsize=(6,4))
df["risk"].value_counts().plot(kind="bar", 
    color=["red","orange","green"])
plt.title("Risk Level Distribution")
plt.xlabel("Risk Level")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig("risk_distribution.png")
plt.show()
print("Chart saved!")

# ─── STEP 5: See averages per risk group ────────
print("\n=== AVERAGES PER RISK GROUP ===")
print(df.groupby("risk")[
    ["attendance","marks1","marks2","assignments"]
].mean().round(1))