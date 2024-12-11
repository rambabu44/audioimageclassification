import os
import shutil
import zipfile
from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel
from typing import List, Optional
from ImageClassification.main import (
    load_and_prepare_data,
    train_model,
    evaluate_model,
    predict_single_image,
)

app = FastAPI()

# In-memory storage for model details and class names
model_store = {"model_path": None, "class_names": None}

UPLOAD_DIR = "uploaded_data"

# Request models
class TrainingRequest(BaseModel):
    data_dir: str
    model_path: Optional[str] = "image_classification_model.pth"
    num_epochs: Optional[int] = 5
    learning_rate: Optional[float] = 0.001


class PredictionRequest(BaseModel):
    image_path: str

@app.post("/upload_data")
async def upload_data_api(file: UploadFile):
    """
    API to upload a zipped folder containing the dataset.
    """
    try:
        # Ensure upload directory exists
        if not os.path.exists(UPLOAD_DIR):
            os.makedirs(UPLOAD_DIR)

        # Save uploaded zip file
        zip_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(zip_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Extract zip file
        extract_dir = os.path.join(UPLOAD_DIR, os.path.splitext(file.filename)[0])
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

        # Remove zip file after extraction
        os.remove(zip_path)

        return {"message": "Dataset uploaded and extracted successfully", "data_dir": extract_dir}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading data: {str(e)}")


@app.post("/load_data")
def load_data_api(data_dir: str):
    """
    API to load and prepare the dataset.
    """
    try:
        train_loader, val_loader, class_names = load_and_prepare_data(data_dir)
        model_store["class_names"] = class_names
        return {
            "message": "Data loaded successfully",
            "num_classes": len(class_names),
            "classes": class_names,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train_model")
def train_model_api(request: TrainingRequest):
    """
    API to train the model with specified parameters.
    """
    try:
        if not model_store.get("class_names"):
            raise HTTPException(status_code=400, detail="Load data first using /load_data API.")

        train_loader, val_loader, class_names = load_and_prepare_data(
            request.data_dir
        )
        model_path = train_model(
            train_loader,
            val_loader,
            class_names,
            num_epochs=request.num_epochs,
            learning_rate=request.learning_rate,
            model_path=request.model_path,
        )
        model_store["model_path"] = model_path
        return {"message": "Model trained successfully", "model_path": model_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate_model")
def evaluate_model_api(data_dir: str):
    """
    API to evaluate the trained model.
    """
    try:
        if not model_store.get("model_path") or not model_store.get("class_names"):
            raise HTTPException(status_code=400, detail="Train the model first using /train_model API.")

        _, val_loader, class_names = load_and_prepare_data(data_dir)
        accuracy = evaluate_model(model_store["model_path"], val_loader, class_names)
        return {"message": "Evaluation completed", "accuracy": accuracy}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
def predict_api(request: PredictionRequest):
    """
    API to predict the class of an image.
    """
    try:
        if not model_store.get("model_path") or not model_store.get("class_names"):
            raise HTTPException(status_code=400, detail="Train the model first using /train_model API.")

        predicted_class = predict_single_image(
            model_store["model_path"], request.image_path, model_store["class_names"]
        )
        return {"message": "Prediction successful", "predicted_class": predicted_class}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
