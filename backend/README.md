cd backend
.venv\Scripts\python.exe -m fastapi dev main.py


cd backend
.venv\Scripts\python.exe -m uvicorn main:app --reload


cd backend
.venv\Scripts\python.exe -m pip install --upgrade fastapi fastapi-cli