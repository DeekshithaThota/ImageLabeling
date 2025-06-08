import torch
import open_clip
from PIL import Image


# Load OpenCLIP model
def load_model():
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    return model, preprocess, tokenizer

def compute_similarity(image, label_prompts, model, preprocess, tokenizer):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Correct way: apply the preprocessing transform to the image
    image_input = preprocess(image).unsqueeze(0).to(device)  # Shape: [1, 3, H, W]

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # Tokenize all labels and encode
        text_inputs = tokenizer(label_prompts).to(device)
        text_features = model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Compute cosine similarity
        similarities = (image_features @ text_features.T).squeeze(0)  # Shape: [num_labels]

    return similarities

