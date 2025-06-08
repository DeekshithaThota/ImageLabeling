import streamlit as st
import os
import json
from PIL import Image
from datetime import datetime
from utils import load_model, compute_similarity
import pandas as pd
import altair as alt
from ultralytics import YOLO  # New import for YOLOv11

st.set_page_config(page_title="OpenCLIP + YOLOv11 App", layout="wide")

@st.cache_resource(show_spinner="Loading OpenCLIP model...")
def get_cached_clip_model():
    return load_model()

@st.cache_resource(show_spinner="Loading YOLOv11 model...")
def get_yolo_model():
    return YOLO("best.pt")  # Make sure best.pt is in your project folder

clip_model, preprocess, tokenizer = get_cached_clip_model()
yolo_model = get_yolo_model()

with open("labels.txt", "r") as f:
    label_prompts = [line.strip() for line in f.readlines()]

st.title("🧠 Image Understanding: OpenCLIP + YOLOv11")

uploaded_file = st.file_uploader("📤 Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        img_width = st.slider("🔧 Adjust image width", 100, 400, 250)
        st.image(image, caption="Uploaded Image", width=img_width)

    with col2:
        # --- OpenCLIP Prediction ---
        similarities = compute_similarity(image, label_prompts, clip_model, preprocess, tokenizer)
        top_idx = similarities.argmax().item()
        predicted_label = label_prompts[top_idx]
        confidence_score = similarities[top_idx].item()

        st.markdown("### 🔍 OpenCLIP Prediction:")
        st.markdown(
            f"<div style='background-color:#DFF0D8;padding:10px;border-radius:10px;font-size:20px'>"
            f"<b>{predicted_label}</b> (Confidence: {confidence_score:.2f})"
            f"</div>",
            unsafe_allow_html=True
        )

        with st.expander("📊 Show All Similarity Scores"):
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

        # Feedback Section
        st.markdown("### 📝 Confirm/Correct label and describe:")
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

            st.success("Image and metadata saved. Thank you!")

    # --- YOLOv11 Detection ---
    st.markdown("## 🕵️ Object Detection using YOLOv11")
    with st.spinner("Running object detection..."):
        results = yolo_model(image)
        results_image_path = "temp_detected.jpg"
        results[0].save(filename=results_image_path)  # Save annotated result image

        st.image(results_image_path, caption="YOLOv11 Detection Output", use_column_width=True)
