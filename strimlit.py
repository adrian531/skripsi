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

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("assets/heart.csv")
    return df

df = load_data()




# Identifikasi kolom kategorikal
categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
# Tambahkan kolom dengan unique values < 10 sebagai kategori (meskipun numerik)
for col in df.columns:
    if col != "HeartDisease" and df[col].nunique() <= 10 and col not in categorical_cols:
        categorical_cols.append(col)

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



# Save model and preprocessors
with open("svc_model.pkl", "wb") as model_file:
    pickle.dump(svc_model, model_file)
with open("encoder.pkl", "wb") as encoder_file:
    pickle.dump(encoder, encoder_file)
with open("scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)


# ---------------- FORM INPUT UNTUK PREDIKSI ---------------- #

# Tentukan kolom numerik asli
original_numerical = [col for col in df.columns if col not in categorical_cols + ['HeartDisease']]

st.subheader("Formulir Prediksi Penyakit Jantung")
with st.form(key='prediction_form'):
    user_input_categorical = {}
    user_input_numerical = {}

    st.markdown("### Input Fitur Kategorikal")
    for col in categorical_cols:
        options = df[col].unique().tolist()
        user_input_categorical[col] = st.selectbox(f"{col}", options)

    st.markdown("### Input Fitur Numerik")
    for col in original_numerical:
        mean_val = df[col].mean()

        if col.lower() == 'oldpeak':  # Hanya Oldpeak yang tetap float
            user_input_numerical[col] = st.number_input(f"{col}", value=float(mean_val), step=0.1, format="%.1f")
        else:
            user_input_numerical[col] = st.number_input(f"{col}", value=int(round(mean_val)), step=1, format="%d")

    submit_button = st.form_submit_button(label='Prediksi')

if submit_button:
    # Buat dataframe dari input user
    input_cat_df = pd.DataFrame([user_input_categorical])
    input_num_df = pd.DataFrame([user_input_numerical])

    # Encode dan gabungkan
    encoded_input = encoder.transform(input_cat_df)
    encoded_input_df = pd.DataFrame(encoded_input, columns=encoder.get_feature_names_out(categorical_cols))
    final_input_df = pd.concat([input_num_df.reset_index(drop=True), encoded_input_df.reset_index(drop=True)], axis=1)

    # Scaling
    input_scaled = scaler.transform(final_input_df)

    # Prediksi
    prediction = svc_model.predict(input_scaled)
    probability = svc_model.predict_proba(input_scaled)[0][1]

    if prediction[0] == 1:
        st.error(f"Pasien berisiko terkena penyakit jantung dengan probabilitas {probability:.2f}")
    else:
        st.success(f"Pasien memiliki risiko rendah terkena penyakit jantung dengan probabilitas {probability:.2f}")
