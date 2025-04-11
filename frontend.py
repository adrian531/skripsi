import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Load model dan preprocessor
with open("svc_model.pkl", "rb") as model_file:
    svc_model = pickle.load(model_file)
with open("encoder.pkl", "rb") as encoder_file:
    encoder = pickle.load(encoder_file)
with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Daftar fitur berdasarkan dataset
feature_info = {
    "age": ("Usia", 29, 100),
    "sex": ("Jenis Kelamin (0: Wanita, 1: Pria)", 0, 1),
    "cp": ("Tipe Nyeri Dada (0-3)", 0, 3),
    "trestbps": ("Tekanan Darah Istirahat (mm Hg)", 80, 200),
    "chol": ("Kolesterol (mg/dl)", 100, 600),
    "fbs": ("Gula Darah Puasa > 120 mg/dl (0: Tidak, 1: Ya)", 0, 1),
    "restecg": ("Hasil Elektrokardiografi (0-2)", 0, 2),
    "thalach": ("Detak Jantung Maksimum", 60, 220),
    "exang": ("Angina Induksi Latihan (0: Tidak, 1: Ya)", 0, 1),
    "oldpeak": ("Depresi ST akibat Latihan", 0.0, 6.2),
    "slope": ("Kemiringan Segmen ST (0-2)", 0, 2),
}

# Judul aplikasi
st.title("Prediksi Penyakit Jantung dengan SVC")

# Form input pengguna
st.subheader("Masukkan Data Pasien")
user_input = []

with st.form("prediction_form"):
    for feature, (label, min_val, max_val) in feature_info.items():
        user_input.append(st.number_input(label, min_value=float(min_val), max_value=float(max_val), value=float((min_val + max_val) / 2)))
    submit_button = st.form_submit_button("Prediksi")

# Proses prediksi saat tombol ditekan
if submit_button:
    user_array = np.array(user_input).reshape(1, -1)
    user_scaled = scaler.transform(user_array)
    prediction = svc_model.predict(user_scaled)
    probability = svc_model.predict_proba(user_scaled)[0][1]

    st.subheader("Hasil Prediksi")
    if prediction[0] == 1:
        st.error(f"Pasien berisiko tinggi terkena penyakit jantung dengan probabilitas {probability:.2f}")
    else:
        st.success(f"Pasien memiliki risiko rendah terkena penyakit jantung dengan probabilitas {probability:.2f}")
