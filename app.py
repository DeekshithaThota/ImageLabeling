import streamlit as st
import os
import json
from PIL import Image
from datetime import datetime
from utils import load_model, compute_similarity
import pandas as pd
import altair as alt

st.set_page_config(page_title="OpenCLIP Similarity App", layout="wide")

@st.cache_resource(show_spinner="Loading OpenCLIP model...")
def get_cached_model():
    return load_model()

model, preprocess, tokenizer = get_cached_model()

# Load labels
with open("labels.txt", "r") as f:
    label_prompts = [line.strip() for line in f.readlines()]

st.title("🔍 Image and Text Similarity using OpenCLIP")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    similarities = compute_similarity(image, label_prompts, model, preprocess, tokenizer)
    top_idx = similarities.argmax().item()  # ✅ convert to plain int
    predicted_label = label_prompts[top_idx]

    

    confidence_score = similarities[top_idx].item()

    st.markdown("### 🔍 Predicted Label:")
    st.markdown(
        f"<div style='background-color:#DFF0D8;padding:10px;border-radius:10px;font-size:20px'>"
        f"<b>{predicted_label}</b> (Confidence: {confidence_score:.2f})"
        f"</div>",
        unsafe_allow_html=True
    )

    with st.expander("📊 Show All Similarity Scores"):
        # Sort similarities descending
        score_data = list(zip(label_prompts, similarities.tolist()))
        score_data.sort(key=lambda x: x[1], reverse=True)

        st.markdown("### 🔢 Sorted Similarity Scores:")
        for label, score in score_data:
            st.markdown(
                f"<div style='padding:6px;font-size:16px;'>"
                f"<b>{label}:</b> {score:.4f}"
                f"</div>",
                unsafe_allow_html=True
            )

        max_n = min(len(score_data), 10)
        top_n = st.slider("Select number of top labels to show in chart", 1, max_n, 5)

        top_scores_df = pd.DataFrame(score_data[:top_n], columns=["Label", "Similarity"])

        chart = (
            alt.Chart(top_scores_df)
            .mark_bar(color="skyblue")
            .encode(
                x=alt.X("Similarity:Q", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("Label:N", sort="-x"),
                tooltip=["Label", "Similarity"]
            )
            .properties(height=top_n * 40)
        )

        st.altair_chart(chart, use_container_width=True)

    # Threshold to decide low confidence
    threshold = 0.20
    if confidence_score < threshold:
        st.warning("Low confidence in prediction. Please help us label this image.")
    else:
        st.success(f"Prediction: **{predicted_label}** (confidence: {confidence_score:.2f})")

    st.markdown("### 📝 Please confirm or correct the label and provide a description:")
    user_label = st.selectbox("Select or confirm a label", label_prompts, index=top_idx)
    user_description = st.text_area("Describe the image")

    if st.button("Submit"):
        os.makedirs("saved_images", exist_ok=True)
        os.makedirs("saved_data", exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_filename = f"{user_label}_{timestamp}.png"
        image_save_path = os.path.join("saved_images", image_filename)
        image.save(image_save_path)

        metadata = {
            "label": user_label,
            "description": user_description,
            "timestamp": timestamp,
            "predicted_label": predicted_label,
            "predicted_confidence": confidence_score,
            "image_filename": image_filename
        }

        metadata_path = os.path.join("saved_data", f"{user_label}_{timestamp}.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

        st.success("Image and metadata saved. Thank you for your help!")
