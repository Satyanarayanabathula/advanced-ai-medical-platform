from app.llm.report_generator import generate_report


def main():
    print("=" * 60)
    print("GROQ LLM REPORT TEST")
    print("=" * 60)

    report = generate_report(
        prediction="Pneumonia",
        probability=0.9311,
        gradcam_available=True,
    )

    print()
    print(report)
    print()
    print("LLM test successful!")


if __name__ == "__main__":
    main()