from __future__ import annotations

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# This model has three NLI classes: entailment, contradiction, and neutral.
MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"


class NLIClassifier:
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
        )
        self.model.to(device)
        self.model.eval()

        self.label_ids = self._find_label_ids()

    def _find_label_ids(self) -> dict[str, int]:
        label2id = {
            str(label).lower().replace("-", "_").replace(" ", "_"): int(label_id)
            for label, label_id in self.model.config.label2id.items()
        }

        aliases = {
            "entailed": ("entailment", "entail"),
            "not entailed": (
                "contradiction",
                "contradict",
                "not_entailment",
                "not_entail",
            ),
            "neutral": ("neutral",),
        }

        found: dict[str, int] = {}
        for output_label, candidates in aliases.items():
            for model_label, label_id in label2id.items():
                if any(candidate in model_label for candidate in candidates):
                    found[output_label] = label_id
                    break

        missing = set(aliases) - set(found)
        if missing:
            available = ", ".join(self.model.config.label2id.keys())
            raise ValueError(
                f"Model '{self.model_name}' is not a three-way NLI model. "
                f"Missing labels: {', '.join(sorted(missing))}. "
                f"Available labels: {available}. "
                "Use a model with entailment, contradiction, and neutral labels, "
                "for example 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'."
            )

        return found

    def classify(self, premise: str, hypothesis: str) -> pd.DataFrame:
        premise = premise.strip()
        hypothesis = hypothesis.strip()

        if not premise:
            raise ValueError("Premise cannot be empty.")
        if not hypothesis:
            raise ValueError("Hypothesis cannot be empty.")

        inputs = self.tokenizer(
            premise,
            hypothesis,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {
            name: tensor.to(self.device)
            for name, tensor in inputs.items()
        }

        with torch.inference_mode():
            logits = self.model(**inputs).logits[0]

        probabilities = torch.softmax(logits, dim=-1)
        rows = [
            {
                "label": label,
                "score": probabilities[label_id].item(),
            }
            for label, label_id in self.label_ids.items()
        ]

        return (
            pd.DataFrame(rows)
            .sort_values("score", ascending=False)
            .reset_index(drop=True)
        )
