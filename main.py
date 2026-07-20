from fastapi import FastAPI, File, UploadFile, HTTPException 
from fastapi.responses import HTMLResponse #return HTML content to the browser instead of JSON
import os #to work with files and folders
import cv2 #decode the img
import numpy as np #convert raw bytes into numpy array
from ocr_utils import extract_text_from_image

app = FastAPI() #starts your FastAPI application

root = os.path.dirname(os.path.abspath(__file__)) #find the html file
static_dir = os.path.join(root, "static")
index_path = os.path.join(static_dir, "index.html")

@app.get("/") #opens page
async def index(): 
    with open(index_path, "r", encoding="utf-8") as f: #opens file in read mode
        html = f.read()
    return HTMLResponse(content=html, status_code=200)

@app.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...)):
    try: #error handling
        contents = await file.read()
        img_arr = np.frombuffer(contents, np.uint8) #bytes to array
        img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image") #checks if decoding worked

        result = extract_text_from_image(img)

        return {
            "filename": file.filename,
            "full_text": result["full_text"],
            "avg_confidence": result["avg_confidence"],
            "detections": result["detections"]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
