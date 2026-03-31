"""
Flask Web Application — Social Network Ads Purchase Predictor
"""

from flask import Flask, request, render_template, redirect, url_for, session
import joblib
import numpy as np

app = Flask(__name__)
app.secret_key = 'sna_secret_key'

# Load model and scaler once at startup
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    # Pull result + inputs from session (set by POST), then clear them
    result = session.pop('result', None)
    age    = session.pop('age', '')
    salary = session.pop('salary', '')
    return render_template('index.html', result=result, age=age, salary=salary)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age    = request.form['age']
        salary = request.form['salary']

        features = np.array([[float(age), float(salary)]])
        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)[0]
        result = "Will Purchase" if prediction == 1 else "Will Not Purchase"

    except Exception as e:
        age    = request.form.get('age', '')
        salary = request.form.get('salary', '')
        result = f"Error: {str(e)}"

    # Store in session then redirect — clears on browser reload
    session['result'] = result
    session['age']    = age
    session['salary'] = salary
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
