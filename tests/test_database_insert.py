from app.database.database import SessionLocal
from app.database.crud import create_prediction


def main():
    db = SessionLocal()

    try:
        prediction = create_prediction(
            db=db,
            filename="test_xray.png",
            prediction="Pneumonia",
            class_id=1,
            probability=0.93,
            normal_probability=0.07,
            pneumonia_probability=0.93,
            gradcam_path="results/gradcam/test_xray.png",
            report="Test AI-assisted medical analysis report.",
            model_version="resnet18-v1",
        )

        print("Prediction inserted successfully.")
        print("Prediction ID:", prediction.id)
        print("Prediction:", prediction.prediction)
        print("Probability:", prediction.probability)

    finally:
        db.close()


if __name__ == "__main__":
    main()