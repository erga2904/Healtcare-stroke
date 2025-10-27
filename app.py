# app.py
import os
import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Load model dan encoder (pastikan file ini ada di root folder)
model = joblib.load('best_model.joblib')
label_encoders = joblib.load('label_encoders.joblib')

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
                # Jika nilai tidak dikenal, gunakan kelas pertama sebagai fallback
                df_input[col] = label_encoders[col].transform([label_encoders[col].classes_[0]])[0]

        # Prediksi
        prediction = model.predict(df_input)[0]
        probability = model.predict_proba(df_input)[0].max()

        result = "Berisiko Stroke" if prediction == 1 else "Tidak Berisiko Stroke"
        confidence = f"{probability * 100:.1f}%"

        return render_template('result.html', result=result, confidence=confidence)

    except Exception as e:
        return f"Error: {str(e)}", 500

# Jalankan aplikasi dengan konfigurasi Railway
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
