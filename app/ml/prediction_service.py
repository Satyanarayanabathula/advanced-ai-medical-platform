import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from app.llm.report_generator import generate_report
from app.ml.gradcam import GradCAM
from app.ml.model import create_model
from app.ml.preprocessing import eval_transform


MODEL_PATH = Path("model/best_resnet18.pth")

CLASS_NAMES = {
    0: "Normal",
    1: "Pneumonia",
}


class PredictionService:

    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = create_model()

        checkpoint = torch.load(
            MODEL_PATH,
            map_location=self.device,
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model = self.model.to(self.device)
        self.model.eval()

    def preprocess_image(self, image: Image.Image):
        if image.mode != "L":
            image = image.convert("L")

        tensor = eval_transform(image)

        tensor = tensor.unsqueeze(0)

        return tensor.to(self.device)

    def predict(self, image: Image.Image):

        image_tensor = self.preprocess_image(image)

        with torch.no_grad():
            output = self.model(image_tensor)

        probabilities = torch.softmax(
            output,
            dim=1,
        )[0]

        predicted_class = torch.argmax(
            probabilities
        ).item()

        probability = probabilities[
            predicted_class
        ].item()

        prediction = CLASS_NAMES[
            predicted_class
        ]

        return {
            "prediction": prediction,
            "class_id": predicted_class,
            "probability": probability,
            "probabilities": {
                "normal": probabilities[0].item(),
                "pneumonia": probabilities[1].item(),
            },
            "input_tensor": image_tensor,
        }

    def generate_gradcam(self, image_tensor):

        target_layer = self.model.layer4[-1]

        gradcam = GradCAM(
            model=self.model,
            target_layer=target_layer,
        )

        heatmap, output, predicted_class = (
            gradcam.generate(image_tensor)
        )

        gradcam.close()

        return {
            "heatmap": heatmap,
            "predicted_class": predicted_class,
        }

    def analyze(self, image: Image.Image):

        prediction_result = self.predict(image)

        gradcam_result = self.generate_gradcam(
            prediction_result["input_tensor"]
        )

        report = generate_report(
            prediction=prediction_result["prediction"],
            probability=prediction_result["probability"],
            gradcam_available=True,
        )

        return {
    "prediction": prediction_result["prediction"],
    "class_id": prediction_result["class_id"],
    "probability": prediction_result["probability"],
    "probabilities": prediction_result["probabilities"],
    "gradcam": gradcam_result["heatmap"],
    "input_tensor": prediction_result["input_tensor"],
    "report": report,
}