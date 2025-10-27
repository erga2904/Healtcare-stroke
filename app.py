# app.py
from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Muat model dan encoder
model = joblib.load('stroke_model.joblib')
label_encoders = joblib.load('label_encoders.joblib')
feature_names = joblib.load('feature_names.joblib')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil data dari form
        data = {
            'gender': request.form['gender'],
            'age': float(request.form['age']),
            'hypertension': int(request.form['hypertension']),
            'heart_disease': int(request.form['heart_disease']),
            'ever_married': request.form['ever_married'],
            'work_type': request.form['work_type'],
            'Residence_type': request.form['Residence_type'],
            'avg_glucose_level': float(request.form['avg_glucose_level']),
            'bmi': float(request.form['bmi']),
            'smoking_status': request.form['smoking_status']
        }

        # Buat DataFrame
        df_input = pd.DataFrame([data])

        # Encode fitur kategorikal
        for col in ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']:
            if df_input[col].iloc[0] in label_encoders[col].classes_:
                df_input[col] = label_encoders[col].transform(df_input[col])
            else:
                # Jika nilai tidak dikenal, gunakan modus (opsional)
                df_input[col] = label_encoders[col].transform([label_encoders[col].classes_[0]])[0]

        # Prediksi
        prediction = model.predict(df_input)[0]
        probability = model.predict_proba(df_input)[0].max()

        result = "Berisiko Stroke" if prediction == 1 else "Tidak Berisiko Stroke"
        confidence = f"{probability * 100:.1f}%"

        return render_template('result.html', result=result, confidence=confidence)

    except Exception as e:
        return f"Error: {str(e)}", 400

if __name__ == '__main__':
    app.run(debug=True)