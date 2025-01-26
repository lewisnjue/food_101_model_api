# Import required libraries
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import torch.nn as nn
import io
import torch
from torchvision import transforms
import os
import pathlib
import uvicorn

# Initialize the FastAPI app
app = FastAPI()

# Set up static files and templates directory
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Define constants
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
MODEL_PATH = os.path.join(BASE_DIR, 'model.pth')  # Path to your trained model

CLASS_NAMES = [
    "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare", 
    "beet_salad", "beignets", "bibimbap", "bread_pudding", "breakfast_burrito", 
    "bruschetta", "caesar_salad", "cannoli", "caprese_salad", "carrot_cake", 
    "ceviche", "cheesecake", "cheese_plate", "chicken_curry", "chicken_quesadilla", 
    "chicken_wings", "chocolate_cake", "chocolate_mousse", "churros", "clam_chowder", 
    "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame", "cup_cakes", 
    "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict", 
    "escargots", "falafel", "filet_mignon", "fish_and_chips", "foie_gras", 
    "french_fries", "french_onion_soup", "french_toast", "fried_calamari", 
    "fried_rice", "frozen_yogurt", "garlic_bread", "gnocchi", "greek_salad", 
    "grilled_cheese_sandwich", "grilled_salmon", "guacamole", "gyoza", "hamburger", 
    "hot_and_sour_soup", "hot_dog", "huevos_rancheros", "hummus", "ice_cream", 
    "lasagna", "lobster_bisque", "lobster_roll_sandwich", "macaroni_and_cheese", 
    "macarons", "miso_soup", "mussels", "nachos", "omelette", "onion_rings", 
    "oysters", "pad_thai", "paella", "pancakes", "panna_cotta", "peking_duck", 
    "pho", "pizza", "pork_chop", "poutine", "prime_rib", "pulled_pork_sandwich", 
    "ramen", "ravioli", "red_velvet_cake", "risotto", "samosa", "sashimi", 
    "scallops", "seaweed_salad", "shrimp_and_grits", "spaghetti_bolognese", 
    "spaghetti_carbonara", "spring_rolls", "steak", "strawberry_shortcake", 
    "sushi", "tacos", "takoyaki", "tiramisu", "tuna_tartare", "waffles"
]



"""  """



class FoodNet(nn.Module):
    def __init__(self, input_shape: int, num_classes: int):
        super(FoodNet, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(input_shape, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.4),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.4),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5)
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x

"""  """
# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(MODEL_PATH, map_location=device)
model = model.to(device)

model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Define routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload/")
async def create_upload_file(file: UploadFile = File(...)):
    try:
        # Read and save the uploaded image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Apply transformations
        input_tensor = transform(image).unsqueeze(0).to(device)

        # Make prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted_idx = torch.max(outputs, 1)
            predicted_class = CLASS_NAMES[predicted_idx.item()]

        return {"filename": file.filename, "food_type": predicted_class, "message": "Prediction successful!"}

    except Exception as e:
        return {"error": str(e), "message": "Something went wrong. Please try again."}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
