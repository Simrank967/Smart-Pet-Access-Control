import streamlit as st
from PIL import Image

from predict import predict_image

# =====================================================
# Page Configuration
# =====================================================

st.set_page_config(
    page_title="Smart Pet Access Control",
    page_icon="🐾",
    layout="wide"
)

# =====================================================
# Sidebar
# =====================================================

with st.sidebar:

    st.title("🐾 Smart Pet Access")

    st.markdown("---")

    st.subheader("About")

    st.write("""
This application uses a **ResNet18 Transfer Learning**
model to classify uploaded pet images as either
a **Dog** or a **Cat**.

Based on the prediction, the smart pet door
either opens or remains locked.
""")

    st.markdown("---")

    st.subheader("Model")

    st.write("✔ ResNet18")
    st.write("✔ Transfer Learning")
    st.write("✔ PyTorch")

    st.markdown("---")

    st.metric("Validation Accuracy", "98.27%")
    st.metric("Test Accuracy", "97.68%")

    st.markdown("---")

    st.caption("Made with ❤️ by Simran Kaur")

# =====================================================
# Header
# =====================================================

st.title("🐾 Smart Pet Access Control System")

st.markdown("""
### AI-Powered Pet Recognition using Deep Learning

Upload an image of a **Cat** or **Dog** to determine whether the smart pet door should **OPEN** or remain **LOCKED**.
""")

st.divider()

# =====================================================
# Image Upload
# =====================================================

uploaded_file = st.file_uploader(
    "📤 Upload an Image",
    type=["jpg", "jpeg", "png"]
)

# =====================================================
# Prediction
# =====================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1.3, 1])

    # ---------------- Image ---------------- #

    with col1:

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    # ---------------- Prediction ---------------- #

    prediction, confidence, door_status = predict_image(image)

    with col2:

        st.subheader("Prediction")

        if prediction == "Dog":

            st.success("🐶 Dog Detected")

        else:

            st.success("🐱 Cat Detected")

        st.metric(
            label="Confidence",
            value=f"{confidence:.2f}%"
        )

        st.progress(confidence / 100)

        st.write("")

        st.subheader("Door Status")

        if door_status == "OPEN":

            st.success("🟢 Door OPEN")

        else:

            st.error("🔴 Door LOCKED")

# =====================================================
# Footer
# =====================================================

st.divider()

st.markdown(
"""
<div style="text-align:center">

<b>Smart Pet Access Control System</b><br>

Built using <b>PyTorch</b> • <b>ResNet18</b> • <b>Transfer Learning</b> • <b>Streamlit</b>

</div>
""",
unsafe_allow_html=True
)