"""
Flask Web Application — Social Network Ads Purchase Predictor
Ensemble: majority vote across LR, KNN, SVM
"""

from flask import Flask, request, render_template, redirect, url_for, session
import joblib
import numpy as np
import os
import csv
from datetime import datetime

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'sna_secret_key_dev')

# Load all 3 models and scaler
lr_model  = joblib.load('model_lr.pkl')
knn_model = joblib.load('model_knn.pkl')
svm_model = joblib.load('model_svm.pkl')
scaler    = joblib.load('scaler.pkl')

LOG_FILE = 'predictions.csv'

def log_prediction(age, salary, lr_vote, knn_vote, svm_vote, result):
    """Append one prediction record to the CSV log."""
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Timestamp', 'Age', 'EstimatedSalary',
                             'LR_Vote', 'KNN_Vote', 'SVM_Vote',
                             'TotalVotes', 'Result'])
        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            age, salary, lr_vote, knn_vote, svm_vote,
            lr_vote + knn_vote + svm_vote, result
        ])

@app.route('/')
def home():
    result = session.pop('result', None)
    age    = session.pop('age', '')
    salary = session.pop('salary', '')
    return render_template('index.html', result=result, age=age, salary=salary)

@app.route('/predict-page')
def predict_page():
    result = session.pop('result', None)
    age    = session.pop('age', '')
    salary = session.pop('salary', '')
    return render_template('predict.html', result=result, age=age, salary=salary)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age    = request.form['age']
        salary = request.form['salary']

        features = scaler.transform(np.array([[float(age), float(salary)]]))

        # Majority vote: 2 or more out of 3 wins
        lr_vote  = int(lr_model.predict(features)[0])
        knn_vote = int(knn_model.predict(features)[0])
        svm_vote = int(svm_model.predict(features)[0])
        votes = lr_vote + knn_vote + svm_vote
        prediction = 1 if votes >= 2 else 0
        result = "Will Purchase" if prediction == 1 else "Will Not Purchase"

        log_prediction(age, salary, lr_vote, knn_vote, svm_vote, result)

    except Exception as e:
        age    = request.form.get('age', '')
        salary = request.form.get('salary', '')
        result = f"Error: {str(e)}"

    session['result'] = result
    session['age']    = age
    session['salary'] = salary
    return redirect(url_for('predict_page'))

if __name__ == '__main__':
    app.run(debug=True)
