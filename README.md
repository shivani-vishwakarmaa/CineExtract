# CineExtract

App to extract structured movie information from a paragraph using LLM. 
<br>
[visit website](https://cineextract.streamlit.app/)

## Files added
- `core.py` — CLI-style extractor (prompts for input).
- `core_ui.py` — Streamlit app UI.
- `requirements.txt` — Python dependencies.

## Setup

1. Create and activate a virtual environment (Windows PowerShell):

```powershell
uv venv
```

2. Install dependencies:

```powershell
uv pip install -r requirements.txt
```

## Run the app

Start the Streamlit UI:

```powershell
streamlit run core_ui.py
```

Or run the CLI extractor:

```powershell
python core.py
```

## Notes
- The project expects API credentials in environment variables — see `.env.example`.


