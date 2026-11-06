from __future__ import annotations

import streamlit as st

from src.classifier import MODEL_NAME, NLIClassifier
from src.device import available_devices


@st.cache_resource(show_spinner="Loading the NLI model...")
def load_classifier(
    model_name: str,
    device: str,
) -> NLIClassifier:
    return NLIClassifier(
        model_name=model_name,
        device=device,
    )


def run_app() -> None:
    st.set_page_config(
        page_title="NLI Entailment Classifier",
        page_icon="🔎",
    )

    st.title("Natural Language Inference")
    st.write(
        "Enter a premise and a hypothesis. The model predicts whether the "
        "hypothesis is entailed, not entailed, or neutral."
    )

    model_name = st.text_input(
        "Hugging Face model name or local path",
        value=MODEL_NAME,
        help=(
            "Use a three-way NLI model with entailment, contradiction, "
            "and neutral labels."
        ),
    ).strip()

    device = st.selectbox(
        "Device",
        available_devices(),
    )

    premise = st.text_area(
        "Premise",
        height=160,
        placeholder="Example: A woman is playing a violin on a stage.",
    )

    hypothesis = st.text_area(
        "Hypothesis",
        height=160,
        placeholder="Example: A person is performing music.",
    )

    if st.button("Evaluate", type="primary"):
        if not model_name:
            st.warning("Enter a model name or local path.")
            return

        if not premise.strip():
            st.warning("Enter a premise.")
            return

        if not hypothesis.strip():
            st.warning("Enter a hypothesis.")
            return

        try:
            classifier = load_classifier(
                model_name,
                device,
            )
            results = classifier.classify(
                premise,
                hypothesis,
            )
            prediction = results.iloc[0]["label"]

            st.success(f"Prediction: {prediction}")

        except Exception as error:
            st.exception(error)