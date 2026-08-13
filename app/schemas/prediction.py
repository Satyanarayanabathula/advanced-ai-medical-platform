from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    id: int

    prediction: str
    class_id: int

    probability: float = Field(
        ge=0.0,
        le=1.0,
    )

    normal_probability: float = Field(
        ge=0.0,
        le=1.0,
    )

    pneumonia_probability: float = Field(
        ge=0.0,
        le=1.0,
    )

    gradcam_available: bool

    gradcam_url: str | None

    report: str