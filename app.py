from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('heart_disease_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Collect form data
        data = [
            int(request.form['age']),
            1 if request.form['gender'] == 'Male' else 0,
            int(request.form['cholesterol']),
            int(request.form['bp']),
            int(request.form['heart_rate']),
            int(request.form['smoking']),
            int(request.form['alcohol']),
            int(request.form['exercise_hours']),
            int(request.form['family_history']),
            int(request.form['diabetes']),
            int(request.form['obesity']),
            int(request.form['stress_level']),
            int(request.form['blood_sugar']),
            int(request.form['angina']),
            int(request.form['chest_pain'])
        ]
        
        # Predict using the model
        prediction = model.predict([np.array(data)])
        result = "Positive for Heart Disease" if prediction == 1 else "Negative for Heart Disease"
        
        return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
