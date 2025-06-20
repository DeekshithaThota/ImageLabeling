import streamlit as st
import os
from PIL import Image
from utils import load_model, compute_similarity
import pandas as pd
import torch.nn.functional as F
import torch

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
        #similarities_softmax = F.softmax(similarities, dim = 0)
        top_idx = similarities.argmax().item()
        confidence_score = similarities[top_idx].item()
        predicted_category = classify_index(top_idx, confidence_score)

        

        # Prepare scores by category
        score_data = list(zip(label_prompts, similarities.tolist()))
        
        vehicle_scores = [('vehicle', label_prompts[i], similarities[i].item()) for i in vehicle_range]
        vehicle_scores = sorted(vehicle_scores, key = lambda x : x[2], reverse= True )
        document_scores = [('document', label_prompts[i], similarities[i].item()) for i in document_range]
        document_scores = sorted(document_scores, key = lambda x : x[2], reverse= True )
        other_scores = [('other', label_prompts[i], similarities[i].item()) for i in other_range]
        other_scores = sorted(other_scores, key = lambda x : x[2], reverse= True )

        

        top_vehicle = sorted(vehicle_scores, key = lambda x : x[2], reverse= True )[:1]
        top_document = sorted(document_scores, key = lambda x : x[2], reverse= True )[:1]
        top_other = sorted(other_scores, key = lambda x : x[2], reverse= True )[:1]

        # Step 2: Combine and apply softmax on top similarities
        combined_top = top_vehicle + top_document + top_other
        top_similarities = torch.tensor([entry[2] for entry in combined_top])
        top_softmax = F.softmax(top_similarities * 100, dim=0).tolist()
        
        # Step 3: Get predicted category from softmax
        max_softmax = max(top_softmax)
        max_index = top_softmax.index(max_softmax)
        predicted_category = combined_top[max_index][0] if max_softmax >= 0.80 else "ambiguous"

        st.markdown("### 🔍 Prediction Result:")
        st.markdown(
            f"<div style='background-color:#E0F7FA;padding:12px;border-radius:10px;font-size:20px;'>"
            f"<b>Predicted Category:</b> {predicted_category.title()}<br>"
            f"</div>",
            unsafe_allow_html=True
        )

        # --- Show Similarities ---
        st.markdown("### 📊 Show Category-wise Similarities")
        # Updated render_score_table to optionally show softmax
        def render_score_table(score_list, title, show_softmax=False):
            df = pd.DataFrame(score_list, columns=["Category", "Prompt", "Similarity"])
            if show_softmax:
                df["Softmax"] = df["Similarity"].apply(lambda x: round(x, 4))  # placeholder, replaced later
            st.markdown(f"**Top {title} Prompts:**")
            st.dataframe(df, use_container_width=True)
        
        with st.expander("Top Prompt from Each Category"):
            # Add softmax to combined_top entries
            combined_top_softmax = [
                (entry[0], entry[1], entry[2], round(softmax_score, 4))
                for entry, softmax_score in zip(combined_top, top_softmax)
            ]
            
            df = pd.DataFrame(combined_top_softmax, columns=["Category", "Prompt", "Similarity", "Softmax"])
            st.dataframe(df, use_container_width=True)


        with st.expander("🚘 Vehicle Similarities"):
            render_score_table(vehicle_scores, "Vehicle")

        with st.expander("📄 Document Similarities"):
            render_score_table(document_scores, "Document")

        with st.expander("🌀 Others Similarities"):
            render_score_table(other_scores, "Others")
