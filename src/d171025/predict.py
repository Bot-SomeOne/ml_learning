import torch
from torchvision import models, transforms
from PIL import Image
import os

# Config
IMG_SIZE = 224  
MODEL_OUT = "catdog_resnet18_best.pt" 
class_names = ["cat", "dog"] 

def load_model():
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load(MODEL_OUT, map_location=device)["model_state"])
    model = model.to(device)
    model.eval()
    return model, device

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    return image

def predict_image(model, device, image_path):
    image = preprocess_image(image_path).to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return class_names[predicted.item()]

if __name__ == "__main__":
    model, device = load_model()
    # image_path = "predict.jpg"  
    image_path = "predict_2.jpg"  
    result = predict_image(model, device, image_path)
    print(f"The image is a: {result}")