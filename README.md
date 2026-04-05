#  Student Performance Predictor
### PS-05 | AI & EdTech Hackathon Project

## 📌 Problem Statement
Teachers find it hard to identify at-risk students 
before exams. Manual tracking is inefficient and 
limits timely support.

## 💡 Solution
An AI-powered web app that predicts student risk 
levels using Machine Learning based on:
- Attendance percentage
- Subject 1 Marks
- Subject 2 Marks
- Assignment Scores

## 🚀 Features
- ⚡ Real ML Model (Random Forest)
- 📊 Live Class Dashboard
- 🔴 Risk Detection (High / Medium / Low)
- 💡 Personalized Improvement Suggestions
- 📋 Class Overview Table with all students

## 🛠️ Tech Stack
- **Frontend** → HTML, CSS, JavaScript
- **Backend** → Python, Flask
- **ML Model** → Scikit-learn (Random Forest)
- **Data** → Pandas, NumPy

## ▶️ How to Run

### Step 1 - Install requirements
pip install flask pandas scikit-learn numpy

### Step 2 - Create dataset
python create_dataset.py

### Step 3 - Train ML model
python train_model.py

### Step 4 - Run the app
python app.py

### Step 5 - Open browser
http://localhost:5000

## 📁 Project Structure
StudentPerdictor/
├── app.py
├── create_dataset.py
├── train_model.py
├── students.csv
├── README.md
└── templates/
    └── index.html

