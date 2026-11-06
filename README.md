# NLI Entailment App

A Streamlit app for three-way natural language inference:

- `entailed`: the hypothesis follows from the premise
- `not entailed`: the hypothesis contradicts the premise
- `neutral`: the premise does not provide enough information

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Test

```bash
pytest -m "not integration"
pytest -m integration
```

The first model load downloads model files from Hugging Face unless a local model path is used.
