# Face Detection API

<img src="https://img.shields.io/badge/Python-3.8%2B-blue" alt="Python 3.8+"> <img src="https://img.shields.io/badge/FastAPI-0.68+-green" alt="FastAPI 0.68+"> <img src="https://img.shields.io/badge/OpenCV-4.5+-red" alt="OpenCV 4.5+">

Сервис для детекции лиц на изображениях

## 📌 Возможности

- 🖼️ Поддержка форматов: JPEG, PNG, WEBP, BMP
- 🔄 Автокоррекция ориентации (EXIF)
- 🔍 3 уровня детекции:
  - **Базовая** (оптимальные параметры)
  - **Повышенная чувствительность** (уменьшенные требования)
  - **Детекция на увеличенном изображении**
- 📊 Возвращает координаты найденных лиц
- ⚠️ Заглушка для проверки на знаменитостей (in progress)

## 🚀 Быстрый старт

### Установка

```bash
pip install -r requirements.txt
```
### Запуск тестового сервера
```bash
uvicorn main:app --reload
```
или
```
python main.py
```
### 📡 API Endpoints
```POST /check_face```

Параметры: 

```file: Изображение (обязательно)```

## Пример запроса:

```bash
curl -X POST -F "file=@photo.jpg" http://localhost:8000/check_face
```
Успешный ответ (200 OK):

```
json
{
  "status": "success",
  "face_found": true,
  "face_location": {
    "x": 100,
    "y": 150,
    "width": 200,
    "height": 200
  },
  "is_celebrity": false
}
```

## ⚠️ Возможные ошибки

| Код | Сообщение               | Описание                     |
|:---:|-------------------------|------------------------------|
| 400 | Фото не передано!       | Не был прикреплен файл с изображением |
| 400 | Нет лица на фото!       | Алгоритм не обнаружил лиц на изображении |
| 500 | Internal server error   | Внутренняя ошибка сервера, подробности в логах |
