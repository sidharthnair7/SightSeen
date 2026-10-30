# Starts SightSeen's 3D helper on http://127.0.0.1:8001. Spring Boot calls it; leave this window open.
Set-Location $PSScriptRoot
& .\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8001
