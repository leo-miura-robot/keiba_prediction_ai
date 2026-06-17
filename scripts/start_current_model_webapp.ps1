param(
  [string]$ConfigPath = "config\current_model_webapp_mvp_v1.yaml",
  [int]$Port = 8501
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
  throw "python was not found on PATH."
}

@'
import importlib
missing = []
for name in ["streamlit", "pandas", "plotly", "pyarrow", "yaml"]:
    try:
        importlib.import_module(name)
    except Exception:
        missing.append(name)
if missing:
    raise SystemExit("Missing Python modules: " + ", ".join(missing))
'@ | python -

if (-not (Test-Path $ConfigPath)) {
  throw "Config not found: $ConfigPath"
}

streamlit run webapp\app.py --server.port $Port
