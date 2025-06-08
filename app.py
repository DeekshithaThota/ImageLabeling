import streamlit as st
import os
import json
from PIL import Image
from datetime import datetime
from utils import load_model, compute_similarity

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
    st.image(image, caption="Uploaded Image", use_column_width=True)

    similarity_scores = compute_similarity(model, tokenizer, preprocess, image, label_prompts)
    
    # Show results
    best_score = max(similarity_scores)
    best_label = label_prompts[similarity_scores.index(best_score)]

    st.write("### 📊 Similarity Scores")
    for label, score in zip(label_prompts, similarity_scores):
        st.write(f"**{label}**: {score:.2f}")

    if best_score < 40:  # Threshold, you can tune
        st.warning("❌ Low confidence in prediction. Please help us label this image.")
        
        user_description = st.text_input("Describe the image:")
        user_label = st.selectbox("Choose the best label:", label_prompts)

        if st.button("Submit"):
            # Save the image and metadata
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_save_path = os.path.join("saved_data", f"{user_label}_{timestamp}.png")
            image.save(image_save_path)

            metadata = {
                "label": user_label,
                "description": user_description,
                "filename": os.path.basename(image_save_path),
                "timestamp": timestamp
            }

            with open(image_save_path.replace(".png", ".json"), "w") as f:
                json.dump(metadata, f)

            st.success("✅ Data saved. Thank you for your help!")
    else:
        st.success(f"✅ Prediction: **{best_label}** (confidence: {best_score:.2f})")
