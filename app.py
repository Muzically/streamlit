import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ==========================
# Konfigurasi
# ==========================
IMG_SIZE = (180, 180)

# Sesuaikan jika urutan kelas berbeda
CLASS_NAMES = [
    "Normal",
    "Pneumonia"
]

st.set_page_config(
    page_title="Streamlit",
    layout="centered"
)

# ==========================
# Load Model
# ==========================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        "model/model_cnn_xray_final.keras"
    )
    return model


model = load_model()

# ==========================
# Preprocessing
# ==========================
def preprocess(image):

    image = image.resize(IMG_SIZE)

    image = np.array(image, dtype=np.float32)

    # Tidak perlu /255
    # karena model sudah memiliki layer Rescaling(1./255)

    image = np.expand_dims(image, axis=0)

    return image


# ==========================
# Tampilan
# ==========================
st.title("Streamlit - Prediksi Rekognisi Data X-Ray Dengan Model Keras")
st.write(
    "Upload gambar X-Ray dada."
)

uploaded = st.file_uploader(
    "Upload Gambar X-Ray",
    type=["jpg", "jpeg", "png"]
)

if uploaded is not None:

    image = Image.open(uploaded).convert("RGB")

    st.image(
        image,
        caption="Gambar yang diupload",
        use_container_width=True
    )

    with st.spinner("Melakukan prediksi..."):

        x = preprocess(image)

        pred = model.predict(x, verbose=0)

        probability = float(pred[0][0])

        if probability >= 0.5:
            predicted_class = CLASS_NAMES[1]
            confidence = probability
        else:
            predicted_class = CLASS_NAMES[0]
            confidence = 1 - probability

    st.divider()

    st.subheader("Hasil Prediksi")

    st.success(f"**{predicted_class}**")

    st.metric(
        label="Confidence",
        value=f"{confidence*100:.2f}%"
    )

    st.progress(confidence)

    st.subheader("Probabilitas")

    st.write(f"**Normal : {(1-probability)*100:.2f}%**")
    st.write(f"**Pneumonia : {probability*100:.2f}%**")