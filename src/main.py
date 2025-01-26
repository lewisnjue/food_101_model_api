from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
import torch
from torchvision import transforms
import uvicorn
import os

# Initialize the FastAPI app
app = FastAPI()

# Load the trained model
MODEL_PATH = "resnet50_food_classifier.pth"  # Path to your trained model
CLASS_NAMES = ["Class_1", "Class_2", ..., "Class_101"]  # Replace with your actual class names

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(MODEL_PATH, map_location=device)
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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
