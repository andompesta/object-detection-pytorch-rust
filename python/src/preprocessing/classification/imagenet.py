from PIL import Image
from torchvision import transforms
import torch

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def preprocess_image(
        filename: str
) -> torch.Tensor:
    input_image = Image.open(filename)
    input_tensor = preprocess(input_image)
    return input_tensor.unsqueeze(0)