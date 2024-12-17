import torch
from torchvision import models, transforms
from PIL import Image
import io

# Проверка доступности устройства
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Настройка модели
num_classes = 4
class_labels = ["Coccidiosis", "Healthy", "New Castle Disease", "Salmonella"]

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# Загрузка весов модели с явным указанием CPU
model.load_state_dict(torch.load("best.pt", map_location=torch.device('cpu'))['model_state_dict'])

# Перемещаем модель на доступное устройство
model.to(device)
model.eval()

# Предобработка изображений
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image = data_transforms(image).unsqueeze(0).to(device)  # Преобразование изображения
    with torch.no_grad():
        outputs = model(image)  # Предсказание
        _, predicted = torch.max(outputs, 1)
    return class_labels[predicted.item()]
