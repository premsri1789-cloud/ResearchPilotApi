# Use the official Python 3.11 slim image to keep the container small
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required for PyMuPDF and general building
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy your requirements file first to leverage Docker layer caching
COPY Requirements.txt .

# Install your Python libraries
RUN pip install --no-cache-dir -r Requirements.txt

# Copy all your Python files into the container
COPY . .

# Expose port 8080 (Cloud Run specifically looks for this port by default)
EXPOSE 8080

# The command to start your FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]