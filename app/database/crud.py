from sqlalchemy.orm import Session

from app.database.models import Prediction


def create_prediction(
    db: Session,
    filename: str,
    prediction: str,
    class_id: int,
    probability: float,
    normal_probability: float,
    pneumonia_probability: float,
    gradcam_path: str | None,
    report: str,
    model_version: str = "resnet18-v1",
):
    record = Prediction(
        filename=filename,
        prediction=prediction,
        class_id=class_id,
        probability=probability,
        normal_probability=normal_probability,
        pneumonia_probability=pneumonia_probability,
        gradcam_path=gradcam_path,
        report=report,
        model_version=model_version,
    )

    db.add(record)
    db.commit()
    db.refresh(record)

    return record


def get_predictions(
    db: Session,
    skip: int = 0,
    limit: int = 50,
):
    return (
        db.query(Prediction)
        .order_by(Prediction.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


def get_prediction(
    db: Session,
    prediction_id: int,
):
    return (
        db.query(Prediction)
        .filter(Prediction.id == prediction_id)
        .first()
    )