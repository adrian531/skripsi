import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from streamlit.components.v1 import html

st.title("🫀 Aplikasi Prediksi Penyakit Jantung")

# Switch untuk menampilkan animasi 3D
show_3d = st.toggle("Tampilkan Animasi 3D Jantung", value=True)

if show_3d:
    st.markdown("#### Visualisasi 3D Jantung")
    html(
        """
       <iframe title="3D Heart Model" frameborder="0" allowfullscreen mozallowfullscreen="true" webkitallowfullscreen="true"
        allow="autoplay; fullscreen; xr-spatial-tracking" xr-spatial-tracking execution-while-out-of-viewport execution-while-not-rendered web-share
        src="https://sketchfab.com/models/a70c0c47fe4b4bbfabfc8f445365d5a4/embed"
        width="100%" height="400">
        </iframe>
        """,
        height=400,
    )

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("assets/heart_cleaned.csv")
    return df

df = load_data()

st.markdown("Gunakan aplikasi ini untuk memprediksi risiko penyakit jantung berdasarkan data medis pasien.")

# Identifikasi kolom kategorikal
categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
# Tambahkan kolom dengan unique values < 10 sebagai kategori (meskipun numerik)
for col in df.columns:
    if col != "HeartDisease" and df[col].nunique() <= 10 and col not in categorical_cols:
        categorical_cols.append(col)

# Simpan urutan kolom asli tanpa target
feature_columns = [col for col in df.columns if col != 'HeartDisease']

# OneHotEncoding
encoder = OneHotEncoder(sparse_output=False)
encoded_data = encoder.fit_transform(df[categorical_cols])
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols))

# Gabungkan dengan kolom numerik
numeric_df = df.drop(columns=categorical_cols)
final_df = pd.concat([numeric_df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

# Pisahkan fitur dan target
y = final_df['HeartDisease']
X = final_df.drop(columns=['HeartDisease'])

# Normalisasi fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
svc_model = SVC(kernel='linear', probability=True)
svc_model.fit(X_train, y_train)

# Save model dan preprocessors
with open("svc_model.pkl", "wb") as model_file:
    pickle.dump(svc_model, model_file)
with open("encoder.pkl", "wb") as encoder_file:
    pickle.dump(encoder, encoder_file)
with open("scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

# ---------------- FORM INPUT ---------------- #

st.subheader("📋 Formulir Input Data Pasien")
with st.form(key='prediction_form'):
    user_input = {}

    for col in df.columns:
        if col == 'HeartDisease':
            continue
        elif col in categorical_cols:
            options = df[col].unique().tolist()
            user_input[col] = st.selectbox(f"🧬 {col}", options)
        else:
            if col.lower() == 'oldpeak':
                user_input[col] = st.number_input(f"📉 {col}", value=0.0, step=0.1, format="%.1f")
            else:
                label = f"📊 {col} (contoh: 19)" if col.lower() == "age" else f"📊 {col}"
                user_input[col] = st.number_input(label, value=0, step=1, format="%d")


    submit_button = st.form_submit_button(label='🔍 Prediksi')

if submit_button:
    input_df = pd.DataFrame([user_input])

    # Pisahkan lagi: numerik dan kategorikal (untuk encoding)
    input_cat_df = input_df[categorical_cols]
    input_num_df = input_df.drop(columns=categorical_cols)

    # Encoding dan penggabungan sesuai training
    encoded_input = encoder.transform(input_cat_df)
    encoded_input_df = pd.DataFrame(encoded_input, columns=encoder.get_feature_names_out(categorical_cols))

    final_input_df = pd.concat([input_num_df.reset_index(drop=True), encoded_input_df.reset_index(drop=True)], axis=1)

    # Scaling
    input_scaled = scaler.transform(final_input_df)

    # Prediksi
    prediction = svc_model.predict(input_scaled)
    probability = svc_model.predict_proba(input_scaled)[0][1]

    if prediction[0] == 1:
        st.error(f"🚨 Pasien berisiko **tinggi** terkena penyakit jantung (Probabilitas: {probability:.2%})")
    else:
        st.success(f"✅ Pasien berisiko **rendah** terkena penyakit jantung (Probabilitas: {probability:.2%})")
