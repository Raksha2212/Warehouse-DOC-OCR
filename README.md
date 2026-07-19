
# Warehouse Document OCR

A FastAPI-based OCR web app that extracts text from uploaded invoice images and returns the result with confidence scores and bounding boxes.

## Features
- Upload invoice or document images.
- Extract text using EasyOCR.
- Return OCR confidence scores.
- Return bounding boxes for detected text.
- Clean frontend served from FastAPI.

## Tech Stack
- FastAPI
- EasyOCR
- OpenCV
- NumPy
- HTML
- CSS
- JavaScript

## Folder Structure
```text
warehouse-ocr/
  .gitignore
  main.py
  ocr_utils.py
  README.md
  samples/
    sample_invoice.png
  static/
    index.html
```

## How It Works
1. The user uploads an image from the frontend.
2. FastAPI receives the file using `UploadFile`.
3. OpenCV decodes and preprocesses the image.
4. EasyOCR extracts text and detections.
5. The backend returns JSON.
6. The frontend displays the result.

## Setup
```bash
python -m venv venv
venv\Scripts\activate
pip install fastapi uvicorn opencv-python easyocr numpy python-multipart
```

## Run
```bash
python -m uvicorn main:app --reload
```

Open:
```text
http://127.0.0.1:8000/
```

## Usage
1. Open the website.
2. Choose an invoice image.
3. Click **Upload & OCR**.
4. View the extracted text and detections.

## Sample Output
Add a screenshot of the app here.

## Future Improvements
- Add image preview before upload.
- Add drag-and-drop support.
- Highlight detected text on the image.
- Add downloadable JSON output.
- Improve OCR preprocessing for difficult scans.
