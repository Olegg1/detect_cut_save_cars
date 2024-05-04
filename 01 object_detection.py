import datetime
from ultralytics import YOLO
import cv2
from helper import create_video_writer


# Определение констант
CONFIDENCE_THRESHOLD = 0.2
GREEN = (0, 255, 0)

# Создаем экземпляр класса для загрузки видео-файла
video_cap = cv2.VideoCapture("v2.mp4")
# Создаем экземпляр класса для записи видео-файла
writer = create_video_writer(video_cap, "v2-out.mp4")

# Загружаем ранее оттренерованную модель YOLO8
model = YOLO("yolov8n.pt")


while True:
    # Метка времени старта обработки
    start = datetime.datetime.now()

    # Считываем очередной кадр из входного видео-потока
    ret, frame = video_cap.read()

    # Если кадров больше не осталось - выходим из цикла
    if not ret:
        break

    # Загружаем кадр в модель YOLO и находим объекты
    detections = model(frame)[0]

    # Перебираем все найденные в кадре объекты
    for data in detections.boxes.data.tolist():
        # Извлекаем коэффициент достоверности, связанный с объектом
        confidence = data[4]

        # Если коэффициент достоверности меньше заданного, то пропускаем этот объект
        if float(confidence) < CONFIDENCE_THRESHOLD:
            continue

        # Рисуем рамку вокруг объекта
        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        cv2.rectangle(frame, (xmin, ymin) , (xmax, ymax), GREEN, 2)

    # Метка времени завершения обработки
    end = datetime.datetime.now()
    # Рассчитываем время, затраченное на обнаружение, и выводим этот показатель в консоль
    total = (end - start).total_seconds()
    print(f"Time to process 1 frame: {total * 1000:.0f} milliseconds")

    # Пересчитываем время в параметр FPS
    fps = f"FPS: {1 / total:.2f}"
    cv2.putText(frame, fps, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)

    # Отображаем кадр на экране
    cv2.imshow("Frame", frame)
    # Записываем кадр в выходной файл
    # Выходим, если нажали q
    if cv2.waitKey(1) == ord("q"):
        break

video_cap.release()
writer.release()
cv2.destroyAllWindows()