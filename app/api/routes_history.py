from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database.crud import get_prediction, get_predictions
from app.database.database import get_db


router = APIRouter(
    tags=["History"],
)


class PredictionHistoryResponse(BaseModel):
    id: int
    filename: str
    prediction: str
    class_id: int
    probability: float
    normal_probability: float
    pneumonia_probability: float
    gradcam_path: str | None
    report: str
    model_version: str
    created_at: str


@router.get(
    "/history",
    response_model=list[PredictionHistoryResponse],
)
def history(
    db: Session = Depends(get_db),
):
    records = get_predictions(db)

    return [
        PredictionHistoryResponse(
            id=record.id,
            filename=record.filename,
            prediction=record.prediction,
            class_id=record.class_id,
            probability=record.probability,
            normal_probability=record.normal_probability,
            pneumonia_probability=record.pneumonia_probability,
            gradcam_path=record.gradcam_path,
            report=record.report,
            model_version=record.model_version,
            created_at=record.created_at.isoformat(),
        )
        for record in records
    ]


@router.get(
    "/prediction/{prediction_id}",
    response_model=PredictionHistoryResponse,
)
def prediction_detail(
    prediction_id: int,
    db: Session = Depends(get_db),
):
    record = get_prediction(
        db,
        prediction_id,
    )

    if record is None:
        raise HTTPException(
            status_code=404,
            detail="Prediction not found.",
        )

    return PredictionHistoryResponse(
        id=record.id,
        filename=record.filename,
        prediction=record.prediction,
        class_id=record.class_id,
        probability=record.probability,
        normal_probability=record.normal_probability,
        pneumonia_probability=record.pneumonia_probability,
        gradcam_path=record.gradcam_path,
        report=record.report,
        model_version=record.model_version,
        created_at=record.created_at.isoformat(),
    )