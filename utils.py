import torch
import open_clip
from PIL import Image
from torchvision import transforms

# Load OpenCLIP model
def load_model():
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    return model, preprocess, tokenizer

# Compute similarity
def compute_similarity(model, tokenizer, preprocess, image, texts):
    image_input = preprocess(image).unsqueeze(0)
    text_tokens = tokenizer(texts)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_tokens)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = (100.0 * image_features @ text_features.T).squeeze(0)
    return similarity.tolist()
