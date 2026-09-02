FROM python:3.12-slim
WORKDIR /app
# Asume que copias/tienes un requirements.txt dentro de ./backend
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Ajusta este comando si usas FastAPI/Uvicorn en lugar de python puro
CMD ["python", "main.py"]