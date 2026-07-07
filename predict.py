import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

model = models.resnet18(weights=None)

num_features = model.fc.in_features
model.fc = nn.Linear(num_features,1)

model.load_state_dict(
    torch.load(
        "models/best_model.pth",
        map_location=device
    )
)

model.to(device)
model.eval()


def predict_image(image):

    image = image.convert("RGB")

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(image_tensor)

        probability = torch.sigmoid(output).item()

        if probability >= 0.5:

            prediction = "Dog"
            confidence = probability * 100
            door_status = "OPEN"

        else:

            prediction = "Cat"
            confidence = (1-probability)*100
            door_status = "LOCKED"

    return prediction, confidence, door_status