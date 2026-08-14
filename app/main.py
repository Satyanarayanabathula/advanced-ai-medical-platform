from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes_history import router as history_router
from app.api.routes_prediction import router as prediction_router
from app.database.database import Base, engine
from app.database.models import Prediction

Base.metadata.create_all(bind=engine)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FILES_DIR = PROJECT_ROOT / "results"

FILES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


app = FastAPI(
    title="Advanced AI Medical Intelligence Platform",
    description=(
        "AI-assisted chest X-ray classification API "
        "with Grad-CAM explainability and LLM report generation."
    ),
    version="1.0.0",
)


app.mount(
    "/files",
    StaticFiles(directory=FILES_DIR),
    name="files",
)


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


app.include_router(
    prediction_router
)

app.include_router(
    history_router
)