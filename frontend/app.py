import requests
import streamlit as st
import os

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000",
)


st.set_page_config(
    page_title="AI Medical Intelligence",
    page_icon="🩺",
    layout="wide",
)


st.title("🩺 Advanced AI Medical Intelligence Platform")

st.write(
    "AI-assisted chest X-ray classification with "
    "Grad-CAM explainability and LLM-generated reporting."
)
st.warning(
    "Research/educational prototype only. "
    "This application is not a medical diagnosis and "
    "must not be used for clinical decision-making."
)

st.info(
    """
    **Model scope**

    This model is trained specifically for binary
    chest X-ray classification:

    - Normal
    - Pneumonia

    It is not trained or validated to detect other
    conditions such as cardiomegaly, fractures,
    cancer, tuberculosis, or other abnormalities.

    Images outside this scope may produce unreliable
    predictions.
    """
)


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------

with st.sidebar:
    st.header("System")

    try:
        response = requests.get(
            f"{API_URL}/health",
            timeout=5,
        )

        if response.ok:
            st.success("Backend: Online")
        else:
            st.error("Backend: Unavailable")

    except requests.RequestException:
        st.error("Backend: Offline")


# ------------------------------------------------------------
# Upload
# ------------------------------------------------------------

st.subheader("Upload Chest X-ray")

uploaded_file = st.file_uploader(
    "Choose an X-ray image",
    type=["png", "jpg", "jpeg"],
)


if uploaded_file is not None:

    st.image(
        uploaded_file,
        caption="Uploaded X-ray",
        use_container_width=True,
    )

    if st.button(
        "🔍 Analyze X-ray",
        type="primary",
    ):

        uploaded_file.seek(0)

        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type,
            )
        }

        with st.spinner(
            "Analyzing X-ray..."
        ):

            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    files=files,
                    timeout=180,
                )

            except requests.RequestException as exc:
                st.error(
                    f"Could not connect to the backend: {exc}"
                )
                st.stop()

        if response.status_code != 200:
            try:
                detail = response.json()
            except ValueError:
                detail = response.text

            st.error(
                f"Analysis failed: {detail}"
            )

        else:

            result = response.json()

            st.success("Analysis completed.")

            st.caption(
                "Supported task: Normal vs Pneumonia classification."
            )
            st.caption(
                "Grad-CAM shows model attribution. "
                "It is not a clinically validated lesion or disease-location map."
            )
            # ------------------------------------------------
            # Main result
            # ------------------------------------------------

            prediction = result["prediction"]
            probability = result["probability"]

            st.subheader("AI Model Result")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Prediction",
                    prediction,
                )

            with col2:
                st.metric(
                    "Model Probability",
                    f"{probability * 100:.2f}%",
                )
            st.caption(
                "This is the model's output probability,"
                "not a measure of clinical certainty."
            )

            with col3:
                st.metric(
                    "Prediction ID",
                    result["id"],
                )

            # ------------------------------------------------
            # Class probabilities
            # ------------------------------------------------

            st.subheader(
                "Class Probabilities"
            )

            prob_col1, prob_col2 = st.columns(2)

            with prob_col1:
                st.metric(
                    "Normal",
                    f"{result['normal_probability'] * 100:.2f}%",
                )

            with prob_col2:
                st.metric(
                    "Pneumonia",
                    f"{result['pneumonia_probability'] * 100:.2f}%",
                )

            # ------------------------------------------------
            # Grad-CAM
            # ------------------------------------------------

            if result.get(
                "gradcam_url"
            ):

                st.subheader(
                    "Explainable AI — Grad-CAM"
                )

                gradcam_url = (
                    API_URL
                    + result["gradcam_url"]
                )

                try:
                    gradcam_response = requests.get(
                        gradcam_url,
                        timeout=30,
                    )

                    if gradcam_response.ok:

                        st.image(
                            gradcam_response.content,
                            caption=(
                                "Grad-CAM explanation: "
                                "regions contributing to "
                                "the model prediction"
                            ),
                            use_container_width=True,
                        )

                    else:
                        st.warning(
                            "Grad-CAM image could not be loaded."
                        )

                except requests.RequestException:
                    st.warning(
                        "Grad-CAM image could not be loaded."
                    )

            # ------------------------------------------------
            # LLM report
            # ------------------------------------------------

            st.subheader(
                "AI-Assisted Report"
            )

            st.markdown(
                result["report"]
            )

            st.info(
                "Grad-CAM shows model attribution and "
                "is not a clinically validated lesion map."
            )


# ------------------------------------------------------------
# Prediction History
# ------------------------------------------------------------

st.divider()

st.subheader("Prediction History")

if st.button(
    "🔄 Refresh History"
):

    try:
        response = requests.get(
            f"{API_URL}/history",
            timeout=30,
        )

        if response.ok:

            history = response.json()

            if history:

                display_rows = []

                for item in history:

                    display_rows.append(
                        {
                            "ID": item["id"],
                            "Filename": item["filename"],
                            "Prediction": item["prediction"],
                            "Probability": (
                                f"{item['probability'] * 100:.2f}%"
                            ),
                            "Model": item["model_version"],
                            "Created": item["created_at"],
                        }
                    )

                st.dataframe(
                    display_rows,
                    use_container_width=True,
                )

            else:
                st.info(
                    "No prediction history found."
                )

        else:
            st.error(
                "Could not load prediction history."
            )

    except requests.RequestException as exc:
        st.error(
            f"Could not connect to backend: {exc}"
        )