import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq


# ------------------------------------------------------------
# Environment configuration
# ------------------------------------------------------------

load_dotenv()


DEFAULT_MODEL = "llama-3.3-70b-versatile"


def get_groq_api_key() -> str:
    """
    Load the Groq API key.

    Priority:
    1. Docker secret: /run/secrets/groq_api_key
    2. Local environment variable: GROQ_API_KEY

    The actual key is never printed or returned in errors.
    """

    # Docker secret
    secret_path = Path("/run/secrets/groq_api_key")

    if secret_path.exists():
        try:
            key = secret_path.read_text(
                encoding="utf-8"
            ).strip()
        except OSError as exc:
            raise RuntimeError(
                "Unable to read the Groq API secret."
            ) from exc

        if key:
            return key

    # Local development / .env
    key = os.getenv("GROQ_API_KEY")

    if key:
        key = key.strip()

        if key:
            return key

    raise RuntimeError(
        "GROQ_API_KEY is not configured."
    )


def get_groq_model() -> str:
    """
    Return the configured Groq model.
    """

    model = os.getenv(
        "GROQ_MODEL",
        DEFAULT_MODEL,
    )

    model = model.strip()

    return model or DEFAULT_MODEL


def create_client() -> Groq:
    """
    Create and return a Groq client.

    The API key is loaded only when the client is created.
    """

    api_key = get_groq_api_key()

    return Groq(
        api_key=api_key
    )


# ------------------------------------------------------------
# LLM system prompt
# ------------------------------------------------------------

SYSTEM_PROMPT = """
You are an AI-assisted medical report writing assistant.

Your ONLY task is to convert verified output from a medical
image classification system into a clear, careful,
human-readable report.

The image classification model has already produced the
prediction and probability.

You MUST NOT independently diagnose the patient.

You MUST NOT invent:
- symptoms
- clinical history
- laboratory findings
- imaging findings that were not provided
- treatment recommendations
- differential diagnoses
- additional diseases

You may only describe:
- the model prediction
- the model probability
- the availability of Grad-CAM
- the intended interpretation of the AI output
- the limitations of the AI system

The report must clearly state that:
- the output is AI-assisted
- the result is not a medical diagnosis
- the model was designed for Normal versus Pneumonia
  classification
- Grad-CAM is an attribution visualization and not a
  clinically validated lesion or disease-location map
- a qualified healthcare professional must make the final
  clinical interpretation

Do not overstate confidence.

Use the following structure:

1. AI Model Prediction
2. Model Probability
3. Explainability
4. Interpretation of the Model Output
5. Medical Disclaimer

Write in clear professional language.
"""


def _validate_inputs(
    prediction: str,
    probability: float,
    gradcam_available: bool,
) -> None:
    """
    Validate the information passed to the LLM.
    """

    if not prediction:
        raise ValueError(
            "Prediction is required."
        )

    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            "Probability must be between 0 and 1."
        )

    if not isinstance(
        gradcam_available,
        bool,
    ):
        raise ValueError(
            "gradcam_available must be a boolean."
        )


def generate_report(
    prediction: str,
    probability: float,
    gradcam_available: bool = True,
) -> str:
    """
    Generate an AI-assisted report using Groq.

    Parameters
    ----------
    prediction:
        Verified model prediction, e.g. "Normal" or
        "Pneumonia".

    probability:
        Model probability for the predicted class,
        represented as a value between 0 and 1.

    gradcam_available:
        Whether a Grad-CAM explanation was generated.

    Returns
    -------
    str
        AI-assisted medical image analysis report.
    """

    _validate_inputs(
        prediction=prediction,
        probability=probability,
        gradcam_available=gradcam_available,
    )

    client = create_client()

    model = get_groq_model()

    probability_text = (
        f"{probability:.4f}"
    )

    gradcam_text = (
        "Grad-CAM visualization is available."
        if gradcam_available
        else "Grad-CAM visualization is not available."
    )

    user_prompt = f"""
Verified AI classification output:

Prediction: {prediction}
Model probability: {probability_text}
Explainability: {gradcam_text}

Generate a careful AI-assisted medical image
analysis report using ONLY the verified information
above.

Do not add findings that are not present in the
verified output.

The report must not be presented as a medical diagnosis.
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            temperature=0,
        )
    except Exception as exc:
        raise RuntimeError(
            "Groq report generation failed."
        ) from exc

    try:
        report = (
            response.choices[0]
            .message
            .content
        )
    except (
        AttributeError,
        IndexError,
        TypeError,
    ) as exc:
        raise RuntimeError(
            "Groq returned an invalid response."
        ) from exc

    if not report:
        raise RuntimeError(
            "Groq returned an empty report."
        )

    return report.strip()