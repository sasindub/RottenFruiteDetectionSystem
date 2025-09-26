from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import io
from PIL import Image

# load model
model = tf.keras.models.load_model("best_model.h5")
class_names = ['freshapples', 'freshbanana', 'freshoranges',
               'rottenapples', 'rottenbanana', 'rottenoranges']

IMG_SIZE = (128, 128)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    # read file
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    pred_class = class_names[np.argmax(preds)]
    confidence = np.max(preds) * 100

    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": f"Predicted: {pred_class} ({confidence:.2f}%)"
    })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
