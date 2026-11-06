import pytest

from src.classifier import NLIClassifier


@pytest.fixture(scope="module")
def classifier() -> NLIClassifier:
    return NLIClassifier(device="cpu")


@pytest.mark.integration
def test_detects_entailment(classifier: NLIClassifier) -> None:
    results = classifier.classify(
        "A woman is playing a violin on a stage.",
        "A person is performing music.",
    )
    assert results.iloc[0]["label"] == "entailed"


@pytest.mark.integration
def test_detects_contradiction(classifier: NLIClassifier) -> None:
    results = classifier.classify(
        "A woman is playing a violin on a stage.",
        "Nobody is making music.",
    )
    assert results.iloc[0]["label"] == "not entailed"


@pytest.mark.integration
def test_detects_neutral(classifier: NLIClassifier) -> None:
    results = classifier.classify(
        "A woman is playing a violin on a stage.",
        "The woman is a professional musician.",
    )
    assert results.iloc[0]["label"] == "neutral"


def test_rejects_empty_premise(classifier: NLIClassifier) -> None:
    with pytest.raises(ValueError, match="Premise cannot be empty"):
        classifier.classify(" ", "Someone is outside.")
