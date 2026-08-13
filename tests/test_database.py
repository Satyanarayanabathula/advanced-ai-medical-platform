from sqlalchemy import inspect

from app.database.database import engine


def main():
    inspector = inspect(engine)

    tables = inspector.get_table_names()

    print("Database tables:")

    for table in tables:
        print("-", table)


if __name__ == "__main__":
    main()