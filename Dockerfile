FROM python:3.11-slim

# Install system packages
# tesseract-ocr: OCR engine for image text extraction
# tesseract-ocr-eng: English language pack for tesseract
# poppler-utils: PDF utilities (needed for PyMuPDF/fitz)
RUN apt-get update && \
    apt-get install -y tesseract-ocr tesseract-ocr-eng poppler-utils && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy code and install dependencies
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt

# Start the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]