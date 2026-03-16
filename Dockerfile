FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir "setuptools<70" wheel && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Train the model during build so the artifact is baked in
RUN python train_and_save_model.py

# Expose port for FastAPI
EXPOSE 8000

# Run FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
