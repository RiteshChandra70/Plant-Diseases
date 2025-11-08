from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from PIL import Image
import torch
from torchvision import transforms, models
from torchvision.models import EfficientNet_B0_Weights
import torch.nn as nn
import io
import os
from pathlib import Path

app = FastAPI()


templates_dir = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))


if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


model_path = "plant_disease_model.pth"


device = torch.device('cpu')


def get_classes_from_dataset(dataset_dir="dataset/train"):
    if os.path.isdir(dataset_dir):
       
        items = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
        return sorted(items)
    return None

classes = get_classes_from_dataset() or [
    'Tomato__Early_blight', 'Tomato_Late_blight', 'Tomato__healthy'
]


def build_model(num_classes: int):
    
    model = models.efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features
   
    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_features, num_classes)
    )
    return model


model = None
try:
    if os.path.isfile(model_path):
        state = torch.load(model_path, map_location=device)

        
        if not isinstance(state, dict):
            model = state
        else:
            
            state_keys = list(state.keys())
            
            uses_plain_linear = any(k.startswith('classifier.1.') for k in state_keys)

           
            base = models.efficientnet_b0(weights=None)
            num_features = base.classifier[1].in_features

            
            saved_out = None
            for k, v in state.items():
                if k.endswith('classifier.1.weight') or k.endswith('classifier.1.1.weight'):
                    saved_out = v.shape[0]
                    break

            load_out = saved_out or len(classes)
            if uses_plain_linear:
                
                base.classifier[1] = nn.Linear(num_features, load_out)
            else:
               
                base.classifier[1] = nn.Sequential(
                    nn.Dropout(0.4),
                    nn.Linear(num_features, load_out)
                )
            model = base

            
            try:
                model.load_state_dict(state)
            except Exception:
                model.load_state_dict(state, strict=False)


            if load_out != len(classes):
                print(f"⚠ Checkpoint has {load_out} outputs but current classes length is {len(classes)}. Copying {min(load_out, len(classes))} weights.")
                
                if uses_plain_linear:
                    trained_fc = model.classifier[1]
                else:
                    trained_fc = model.classifier[1][1]


                new_linear = nn.Linear(num_features, len(classes))
                with torch.no_grad():
                    ncopy = min(load_out, len(classes))
                    new_linear.weight[:ncopy] = trained_fc.weight[:ncopy]
                    new_linear.bias[:ncopy] = trained_fc.bias[:ncopy]

                model.classifier[1] = nn.Sequential(nn.Dropout(0.4), new_linear)

        model.to(device)
        model.eval()
        print("✅ Model loaded successfully.")
    else:
        print(f"⚠ Model file not found at {model_path}")
        model = None
except Exception as e:
    model = None
    print("⚠ Error loading model:", e)


def transform_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return transform(image).unsqueeze(0)

def get_prediction(image_bytes):
    if model is None:
        return "Model not loaded"
    tensor = transform_image(image_bytes).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        _, predicted = outputs.max(1)
        idx = int(predicted.item())
        if 0 <= idx < len(classes):
            return classes[idx]
        return str(idx)

# 🏠 Home Page
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

# 🔍 Prediction Endpoint
@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    image_bytes = await file.read()
    prediction = get_prediction(image_bytes)
    return templates.TemplateResponse("index.html", {"request": request, "result": prediction})