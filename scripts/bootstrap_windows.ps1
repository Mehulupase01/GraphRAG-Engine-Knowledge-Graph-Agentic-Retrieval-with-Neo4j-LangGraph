$ErrorActionPreference = "Stop"

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
}

$conda = "C:\Users\Mehul-PC\anaconda3\Scripts\conda.exe"
if (-not (Test-Path $conda)) {
    throw "Conda was not found at $conda"
}

& $conda create -y -n RAGenv python=3.11
& $conda run -n RAGenv python -m pip install -e ".[dev,local]"

Write-Host "Bootstrap complete. Activate RAGenv, add your API key to .env if needed, and place PDFs in data/raw/."
