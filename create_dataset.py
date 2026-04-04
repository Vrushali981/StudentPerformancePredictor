import pandas as pd
import random

# Set seed so same data generates every time
random.seed(42)

students = []

for i in range(200):  # Create 200 students
    
    # Randomly generate student data
    attendance  = random.randint(20, 100)
    marks1      = random.randint(15, 100)
    marks2      = random.randint(15, 100)
    assignments = random.randint(20, 100)
    
    avg_marks = (marks1 + marks2) / 2
    
    # Decide risk based on logical rules
    if attendance < 50 or avg_marks < 40:
        risk = "High"
    elif attendance < 70 or avg_marks < 60:
        risk = "Medium"
    else:
        risk = "Low"
    
    students.append({
        "attendance":   attendance,
        "marks1":       marks1,
        "marks2":       marks2,
        "assignments":  assignments,
        "risk":         risk
    })

# Save to CSV file
df = pd.DataFrame(students)
df.to_csv("students.csv", index=False)

print("Dataset created!")
print(df.head(10))         # Show first 10 rows
print("\nShape:", df.shape) # Show rows x columns
print("\nRisk counts:")
print(df["risk"].value_counts()) # Count each category