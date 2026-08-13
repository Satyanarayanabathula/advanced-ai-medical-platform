from app.database.database import Base, engine
from app.database.models import Prediction


def main():
    print("Creating database tables...")

    Base.metadata.create_all(
        bind=engine
    )

    print("Database initialization complete.")


if __name__ == "__main__":
    main()