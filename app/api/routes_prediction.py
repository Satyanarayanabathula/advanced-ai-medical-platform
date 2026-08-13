from io import BytesIO
from pathlib import Path
from uuid import uuid4

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    UploadFile,
)
from PIL import Image, UnidentifiedImageError
from sqlalchemy.orm import Session

from app.database.crud import create_prediction
from app.database.database import get_db
from app.ml.gradcam_visualization import save_gradcam_overlay
from app.ml.prediction_service import PredictionService
from app.schemas.prediction import PredictionResponse


router = APIRouter(
    prefix="/predict",
    tags=["Prediction"],
)

prediction_service = PredictionService()

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GRADCAM_DIR = (
    PROJECT_ROOT
    / "results"
    / "gradcam_api"
)


@router.post(
    "",
    response_model=PredictionResponse,
)
async def predict(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    if not file.content_type:
        raise HTTPException(
            status_code=400,
            detail="Unable to determine file type.",
        )

    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Only image files are allowed.",
        )

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    try:
        image = Image.open(
            BytesIO(content)
        )

        image.load()

    except (
        UnidentifiedImageError,
        OSError,
    ) as exc:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid image.",
        ) from exc

    try:
        result = prediction_service.analyze(
            image
        )

        filename = (
            f"{uuid4().hex}.png"
        )

        output_path = (
            GRADCAM_DIR / filename
        )

        save_gradcam_overlay(
            original_tensor=result[
                "input_tensor"
            ][0],
            heatmap=result["gradcam"],
            output_path=output_path,
        )

        relative_gradcam_path = (
            f"gradcam_api/{filename}"
        )

        record = create_prediction(
            db=db,
            filename=file.filename or "uploaded_image",
            prediction=result["prediction"],
            class_id=result["class_id"],
            probability=result["probability"],
            normal_probability=result[
                "probabilities"
            ]["normal"],
            pneumonia_probability=result[
                "probabilities"
            ]["pneumonia"],
            gradcam_path=relative_gradcam_path,
            report=result["report"],
            model_version="resnet18-v1",
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Image analysis failed.",
        ) from exc

    return PredictionResponse(
        id=record.id,
        prediction=result["prediction"],
        class_id=result["class_id"],
        probability=result["probability"],
        normal_probability=result[
            "probabilities"
        ]["normal"],
        pneumonia_probability=result[
            "probabilities"
        ]["pneumonia"],
        gradcam_available=True,
        gradcam_url=(
            f"/files/{relative_gradcam_path}"
        ),
        report=result["report"],
    )