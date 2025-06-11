import streamlit as st
import os
from PIL import Image
from utils import load_model, compute_similarity
import pandas as pd

# Streamlit setup
st.set_page_config(page_title="Vehicle/Document Classifier", layout="wide")

@st.cache_resource(show_spinner="Loading OpenCLIP model...")
def get_cached_model():
    return load_model()

model, preprocess, tokenizer = get_cached_model()

# Load prompts
with open("labels.txt", "r") as f:
    label_prompts = [line.strip() for line in f.readlines()]

# Define category ranges
vehicle_range = range(0, 24)
document_range = range(24, 27)
other_range = range(27, len(label_prompts))

# Classification logic
def classify_index(idx, score, threshold=0.225):
    if score < threshold:
        return "ambiguous"
    elif idx in vehicle_range:
        return "vehicle"
    elif idx in document_range:
        return "document"
    else:
        return "others"

# Title
st.title("🚘📄 Vehicle/Document/OpenCLIP Classifier")

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
            f"<b>Predicted Category:</b> {predicted_category.title()}<br>"
            f"</div>",
            unsafe_allow_html=True
        )

        # --- Show Similarities ---
        st.markdown("### 📊 Show Category-wise Similarities")

        # Prepare scores by category
        score_data = list(zip(label_prompts, similarities.tolist()))
        
        vehicle_scores = [('vehicle', label_prompts[i], similarities[i].item()) for i in vehicle_range]
        document_scores = [('document', label_prompts[i], similarities[i].item()) for i in document_range]
        other_scores = [('other', label_prompts[i], similarities[i].item()) for i in other_range]

        top_vehicle = sorted(vehicle_scores, key = lambda x : x[1], reverse= True )[:1]
        top_document = sorted(document_scores, key = lambda x : x[1], reverse= True )[:1]
        top_other = sorted(other_scores, key = lambda x : x[1], reverse= True )[:1]

        
        def render_score_table(score_list, title):
            score_list = sorted(score_list, key=lambda x: x[1], reverse=True)
            df = pd.DataFrame(score_list, columns=["Category", "Prompt", "Similarity"])
            st.markdown(f"**Top {title} Prompts:**")
            st.dataframe(df, use_container_width=True)

        with st.expander("Top Prompt from Each Category"):
            combined_top = top_vehicle + top_document + top_other
            render_score_table(combined_top, "Top 1 per Category")

        with st.expander("🚘 Vehicle Similarities"):
            render_score_table(vehicle_scores, "Vehicle")

        with st.expander("📄 Document Similarities"):
            render_score_table(document_scores, "Document")

        with st.expander("🌀 Others Similarities"):
            render_score_table(other_scores, "Others")
