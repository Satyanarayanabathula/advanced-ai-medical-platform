from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database.database import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    filename: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    prediction: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    class_id: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    probability: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    normal_probability: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    pneumonia_probability: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    gradcam_path: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
    )

    report: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    model_version: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        default="resnet18-v1",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )