docker compose up -d qdrant neo4j

$jobs = @(
    @{ cwd = "backend"; cmd = "../.venv/Scripts/python -m uvicorn app.main:app --reload --port 8000" },
    @{ cwd = ".";       cmd = ".venv/Scripts/streamlit run frontend/app.py" }
)

foreach ($j in $jobs) {
    $script = "Set-Location '$((Get-Item $j.cwd).FullName)'; $($j.cmd)"
    Start-Process powershell -ArgumentList "-NoExit", "-Command", $script
}
