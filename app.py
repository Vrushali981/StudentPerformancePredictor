# ════════════════════════════════════════════
# IMPORT LIBRARIES
# ════════════════════════════════════════════
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

# ════════════════════════════════════════════
# CREATE FLASK APP
# ════════════════════════════════════════════
app = Flask(__name__)
# This creates your web server!
# __name__ tells Flask where to find files

# ════════════════════════════════════════════
# LOAD YOUR TRAINED ML MODEL
# ════════════════════════════════════════════
model = pickle.load(open("model.pkl", "rb"))
print("Model loaded successfully! ✅")
# rb = read binary (model.pkl is a binary file)

# ════════════════════════════════════════════
# LOAD STUDENT DATA FOR DASHBOARD
# ════════════════════════════════════════════
df = pd.read_csv("students.csv")

# ════════════════════════════════════════════
# ROUTE 1 — HOME PAGE
# ════════════════════════════════════════════
@app.route("/")
def home():
    # When someone opens your website
    # Flask shows index.html from templates folder
    return render_template("index.html")

# @app.route("/") means:
# "When browser goes to http://localhost:5000/
#  run this function"

# ════════════════════════════════════════════
# ROUTE 2 — PREDICTION API
# ════════════════════════════════════════════
@app.route("/predict", methods=["POST"])
def predict():
    # methods=["POST"] means:
    # this route only accepts data being SENT to it
    # (not just viewing a page)

    try:
        # ── Get data sent from browser ──────────
        data = request.json
        # request.json reads the data frontend sent

        attendance  = float(data["attendance"])
        marks1      = float(data["marks1"])
        marks2      = float(data["marks2"])
        assignments = float(data["assignments"])
        name        = data.get("name", "Student")
        # .get() safely gets value, 
        # uses "Student" if name not provided

        # ── Prepare data for model ──────────────
        input_data = pd.DataFrame([[
            attendance, marks1, marks2, assignments
        ]], columns=["attendance","marks1",
                     "marks2","assignments"])

        # ── Make prediction ─────────────────────
        prediction  = model.predict(input_data)[0]
        # [0] gets first result from array

        # ── Get confidence score ────────────────
        proba       = model.predict_proba(input_data)[0]
        confidence  = round(max(proba) * 100, 1)
        # predict_proba gives probability for each class
        # max() gets highest probability
        # That's the confidence!

        # ── Generate suggestions ────────────────
        suggestions = getSuggestions(
            prediction, attendance, marks1, 
            marks2, assignments
        )

        # ── Send result back to browser ─────────
        return jsonify({
            "success":    True,
            "name":       name,
            "risk":       prediction,
            "confidence": confidence,
            "suggestion": suggestions,
            "avg_marks":  round((marks1+marks2)/2, 1)
        })
        # jsonify converts Python dict to JSON
        # JSON is how web sends data!

    except Exception as e:
        # If something goes wrong, send error message
        return jsonify({
            "success": False,
            "error":   str(e)
        })

# ════════════════════════════════════════════
# ROUTE 3 — GET ALL STUDENTS DATA
# ════════════════════════════════════════════
@app.route("/students", methods=["GET"])
def get_students():
    # Sends all student data to dashboard

    students_list = []

    for _, row in df.iterrows():
        # iterrows() loops through each row

        # Predict risk for each student
        input_data = pd.DataFrame([[
            row["attendance"], row["marks1"],
            row["marks2"],    row["assignments"]
        ]], columns=["attendance","marks1",
                     "marks2","assignments"])

        risk = model.predict(input_data)[0]

        students_list.append({
            "attendance":  int(row["attendance"]),
            "marks1":      int(row["marks1"]),
            "marks2":      int(row["marks2"]),
            "assignments": int(row["assignments"]),
            "risk":        risk,
            "avg_marks":   round(
                (row["marks1"]+row["marks2"])/2, 1
            )
        })

    # Count risk levels
    high   = sum(1 for s in students_list 
                 if s["risk"]=="High")
    medium = sum(1 for s in students_list 
                 if s["risk"]=="Medium")
    low    = sum(1 for s in students_list 
                 if s["risk"]=="Low")

    return jsonify({
        "students": students_list[:20],
        # Send first 20 students only
        "total":    len(students_list),
        "high":     high,
        "medium":   medium,
        "low":      low
    })

# ════════════════════════════════════════════
# SUGGESTION FUNCTION
# ════════════════════════════════════════════
def getSuggestions(risk, att, m1, m2, assign):
    avg = (m1 + m2) / 2
    tips = []

    if risk == "High":
        if att < 50:
            tips.append("📅 Increase attendance immediately")
        if avg < 40:
            tips.append("📚 Revise basic concepts daily")
        if assign < 50:
            tips.append("✏️ Complete all assignments")
        tips.append("👨‍🏫 Meet teacher for extra help")
        return "⚠️ Urgent: " + " · ".join(tips)

    elif risk == "Medium":
        if att < 70:
            tips.append("📅 Improve attendance by 10%")
        if avg < 60:
            tips.append("📖 Practice 2 hours daily")
        tips.append("👥 Join study groups")
        return "💡 Keep Going: " + " · ".join(tips)

    else:
        return "🌟 Excellent! Help your peers and take on challenges!"

# ════════════════════════════════════════════
# RUN THE SERVER
# ════════════════════════════════════════════
if __name__ == "__main__":
    app.run(debug=True)
    # debug=True → shows errors clearly
    # Runs on http://localhost:5000