import streamlit as st
import os
import json
from PIL import Image
from datetime import datetime
from utils import load_model, compute_similarity
import pandas as pd
import altair as alt

st.set_page_config(page_title="Vehicle / Document Classifier", layout="wide")

@st.cache_resource(show_spinner="Loading OpenCLIP model...")
def get_cached_model():
    return load_model()

model, preprocess, tokenizer = get_cached_model()

# Load prompts
with open("labels.txt", "r") as f:
    label_prompts = [line.strip() for line in f.readlines()]

vehicle_range = range(0, 24)  # First 24 are vehicle prompts
document_range = range(24, 27)  # Next 3 are documents
other_range = range(27, len(label_prompts))  # Rest are others

def classify_index(idx, score, threshold=0.225):
    if score < threshold:
        return "ambiguous"
    elif idx in vehicle_range:
        return "vehicle"
    elif idx in document_range:
        return "document"
    else:
        return "others"

st.title("🚘📄 Vehicle/Document/OpenCLIP Image Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(image, caption="Uploaded Image", width=300)

    with col2:
        similarities = compute_similarity(image, label_prompts, model, preprocess, tokenizer)
        top_idx = similarities.argmax().item()
        confidence_score = similarities[top_idx].item()
        predicted_category = classify_index(top_idx, confidence_score)

        st.markdown("### 🔍 Prediction Result:")
        st.markdown(
            f"<div style='background-color:#E0F7FA;padding:12px;border-radius:10px;font-size:20px;'>"
            f"<b>Category:</b> {predicted_category.title()}<br>"
            f"<b>Top Prompt:</b> {label_prompts[top_idx]}<br>"
            f"<b>Confidence:</b> {confidence_score:.3f}"
            f"</div>",
            unsafe_allow_html=True
        )

        with st.expander("📊 Show All Similarity Scores"):
            score_data = list(zip(label_prompts, similarities.tolist()))
            score_data.sort(key=lambda x: x[1], reverse=True)

            max_n = min(len(score_data), 10)
            top_n = st.slider("Select number of top labels to show in chart", 1, max_n, 5)
            top_scores_df = pd.DataFrame(score_data[:top_n], columns=["Label", "Similarity"])

            chart = (
                alt.Chart(top_scores_df)
                .mark_bar(color="steelblue")
                .encode(
                    x=alt.X("Similarity:Q", scale=alt.Scale(domain=[0, 1])),
                    y=alt.Y("Label:N", sort="-x"),
                    tooltip=["Label", "Similarity"]
                )
                .properties(height=top_n * 40)
            )

            st.altair_chart(chart, use_container_width=True)

        # Confirm/correct label
        st.markdown("### 📝 Confirm or Correct Label and Describe:")
        user_label = st.selectbox("Label", ["vehicle", "document", "others", "ambiguous"], index=["vehicle", "document", "others", "ambiguous"].index(predicted_category))
        user_description = st.text_area("Describe this image (optional):")

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
                "predicted_label": predicted_category,
                "predicted_prompt": label_prompts[top_idx],
                "predicted_confidence": confidence_score,
                "image_filename": image_filename
            }

            metadata_path = os.path.join("saved_data", f"{user_label}_{timestamp}.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=4)

            st.success("✅ Image and feedback saved!")

