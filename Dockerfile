# Menggunakan base image Python 3.9
FROM python:3.9

# Mengatur working directory
WORKDIR /app

# Menyalin requirements.txt ke working directory
COPY requirements.txt .

# Menginstal dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Menyalin seluruh konten ke working directory
COPY . .

# Menjalankan FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]