import io
import base64
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
# from ultralytics import YOLO

app = FastAPI(title="Automatic Checkout Demo")

# ==========================================
# KHU VỰC LOAD MODEL
# ==========================================
# Khởi tạo các model YOLO của bạn ở đây.
# Hiện tại để comment lại để có thể chạy server trước khi model sẵn sàng.
# models = {
#     "model_1": YOLO("models/best_yolov8n.pt"),
#     "model_2": YOLO("models/best_yolov8s.pt"),
#     "model_3": YOLO("models/best_yolov9.pt")
# }

# Mount thư mục static để phục vụ file HTML/CSS/JS
app.mount("/static", StaticFiles(directory="demo_app/static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("demo_app/static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/detect")
async def detect_objects(
    file: UploadFile = File(...),
    model_name: str = Form("model_1")
):
    # Đọc ảnh từ request
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    crops = []
    
    # ==========================================
    # KHU VỰC INFERENCE MODEL
    # ==========================================
    # Thay thế phần code giả lập dưới đây bằng code inference thật
    # selected_model = models.get(model_name)
    # results = selected_model(img)
    # for box in results[0].boxes:
    #     x1, y1, x2, y2 = map(int, box.xyxy[0])
    #     conf = float(box.conf[0])
    #     cls = int(box.cls[0])
    #     # Cắt ảnh (Crop)
    #     cropped = img[y1:y2, x1:x2]
    #     # Xử lý base64 ...

    
    # ---------------------------------------------------------
    # CODE GIẢ LẬP NHẬN DIỆN (Demo Placeholder)
    h, w, _ = img.shape
    x1, y1 = int(w * 0.25), int(h * 0.25)
    x2, y2 = int(w * 0.75), int(h * 0.75)
    
    # Vẽ bounding box lên ảnh gốc
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, "Product (Demo)", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Cắt vật thể
    cropped = img[y1:y2, x1:x2]
    
    # Encode ảnh đã crop
    _, crop_buffer = cv2.imencode('.jpg', cropped)
    crop_base64 = base64.b64encode(crop_buffer).decode('utf-8')
    
    crops.append({
        "label": "Product (Demo)",
        "confidence": 0.95,
        "image": f"data:image/jpeg;base64,{crop_base64}"
    })
    # ---------------------------------------------------------
    
    # Encode ảnh gốc đã vẽ box
    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return JSONResponse(content={
        "annotated_image": f"data:image/jpeg;base64,{img_base64}",
        "crops": crops
    })

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
