from fastapi import FastAPI, UploadFile, File, HTTPException
import cv2
import numpy as np
import uvicorn
from PIL import Image, ImageOps
from typing import Tuple, Optional
import io
import logging


app = FastAPI()
logging.basicConfig(level=logging.INFO)

# Конфиг детектора лиц
FACE_SETTINGS = {
    # Насколько уменьшается изображение на каждом шаге сканирования (1.05 = 5%)
    'scale_factor': 1.05,
    # Сколько соседних прямоугольников должны подтвердить обнаружение лица
    'min_neighbors': 5,
    # Min размер обнаруживаемого лица (pxl)
    'min_size': (50, 50)
}


def load_image(file_contents: bytes) -> Tuple[Optional[np.ndarray], str]:
    """ Загрузчик изображений с конвертацией и обработкой ошибок. """
    try:
        # Попытка через PIL
        with Image.open(io.BytesIO(file_contents)) as img:
            # Исправление ориентации (поворот):
            img = ImageOps.exif_transpose(img)
            
            # Конвертация в RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Преобразование в numpy-массив
            img_array = np.array(img)
            
            # Проверка валидности
            if len(img_array.shape) != 3 or img_array.shape[2] != 3:
                return None, "Недопустимые размеры изображения"
                
            # Конвертация в BGR (Blue-Green-Red) для OpenCV
            return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR), ""
    
    except Exception as pil_error:
        logging.warning(
            f"PIL decode error: {str(pil_error)}"
        )

    # Попытка через OpenCV
    try:
        # Перевод бинарных данных в numpy-массив типа uint8
        nparr = np.frombuffer(file_contents, np.uint8)
        # Декодирование numpy-массива в изображение OpenCV в формате BGR
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:
            return img, ""
        return None, "OpenCV decode error."

    except Exception as cv_error:
        logging.error(f"OpenCV decode error: {str(cv_error)}")
        return (
            None,
            "PIL & OpenCV both decode error."
        )


def detect_faces(image: np.ndarray) -> list:
    """Обнаружение лиц с тремя уровнями чувствительности."""
    # Конвертация в gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Улучшение контраста
    gray = cv2.equalizeHist(gray)
    
    # Загрузка каскада
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Уровень 1: Основной детектор
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=FACE_SETTINGS['scale_factor'],
        minNeighbors=FACE_SETTINGS['min_neighbors'],
        minSize=FACE_SETTINGS['min_size'],
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    # Уровень 2: Повышенная чувствительность
    if len(faces) == 0:
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.02,
            minNeighbors=3,
            minSize=(30, 30)
        )
    
    # Уровень 3: Увеличенное изображение
    if len(faces) == 0:
        resized = cv2.resize(gray, None, fx=1.2, fy=1.2)
        faces = face_cascade.detectMultiScale(
            resized,
            scaleFactor=1.05,
            minNeighbors=4
        )
    
    return faces if faces is not None else []


@app.post("/check_face")
async def check_face(file: UploadFile = File(...)):
    """ Основная функция, всё действо здесь. """

    try:
        # Проверка файла
        if not file.filename:
            raise HTTPException(400, "Фото не передано!")
        
        # Чтение содержимого
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(400, "Файл не содержит контента!")
        
        # Загрузка изображения
        image, error_msg = load_image(contents)
        if image is None:
            raise HTTPException(
                400,
                f"Неудачная конвертация изображения. Попробуйте другое фото: {error_msg}"
            )
        
        # Детекция лиц
        faces = detect_faces(image)
        if len(faces) == 0:
            raise HTTPException(400, "Нет лица на фото!")
        
        # Проверка на знаменитость (берём первое лицо, если их > 1)
        x, y, w, h = faces[0]
        face_roi = image[y:y+h, x:x+w]
        
        # result, err = is_celebrity_face(face_roi)
        # if result:
        #     raise HTTPException(400, err)
        
        return {
            "status": "success",
            "face_found": True,
            "face_location": {
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h)
            },
            "is_celebrity": False
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Server error")
        raise HTTPException(500, "Internal server error")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        workers=1  # для production лучше 4 поставить
    )
