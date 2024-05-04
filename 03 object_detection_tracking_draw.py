# Это оригинальный алгоритм DeepSort - https://github.com/nwojke/deep_sort
# Адаптация DeepSort для реального времени - https://github.com/levan92/deep_sort_realtime
# Используем улучшенный алгоритм

import datetime
from ultralytics import YOLO
import cv2
from helper import create_video_writer
from deep_sort_realtime.deepsort_tracker import DeepSort
import time
from time import sleep
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

CONFIDENCE_THRESHOLD = 0.2
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)

# Создаем экземпляр класса для загрузки видео-файла
video_cap = cv2.VideoCapture("C:\\Users\\User\\Desktop\\ML_CV\\DeepSort\\DeepSort\\BMW_MERSEDESS.mp4")
# Создаем экземпляр класса для записи видео-файла
writer = create_video_writer(video_cap, "out.mp4")

# Загружаем ранее оттренерованную модель YOLO8
model = YOLO("C:\\Users\\User\\Desktop\\ML_CV\\DeepSort\\DeepSort\\best (6).pt")
classes = ['bmw', 'mersedess']
# Создаём объект трекера
tracker = DeepSort(max_age=50)
counter = 0
time_counter = 0
while True:
    start = datetime.datetime.now()

    ret, frame = video_cap.read()

    if not ret:
        break

    # Создаем массив для информации о найденных объектах
    results = []

    ######################################
    # Блок определения объектов
    ######################################

    # Загружаем кадр в модель YOLO и находим объекты
    detections = model(frame)[0]

    # Обрабатываем каждый объект
    for data in detections.boxes.data.tolist():
        # Извлекаем коэффициент достоверности, связанный с объектом
        confidence = data[4]

        # Если коэффициент достоверности меньше заданного, то пропускаем этот объект
        if float(confidence) < CONFIDENCE_THRESHOLD:
            continue

        # Рисуем рамку вокруг объекта
        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        class_id = int(data[5])

        # Добавляем координаты найденного объекта (x, y, w, h), коэффициент достоверности и класс в массив
        results.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_id])


    ######################################
    # Трекинг объектов
    ######################################

    # Загружаем все наши найденные объекты в DeepSort
    tracks = tracker.update_tracks(results, frame=frame)

    unique_class_ids = set()
    unique_tracks = []
    for track in tracks[::-1]:
        if not track.is_confirmed():
            continue

        class_id = track.get_det_class()

        if class_id not in unique_class_ids:
            unique_class_ids.add(class_id)
            unique_tracks.append(track)

    for track in unique_tracks:
        if len(results) < 1: continue
        # Если трэк не достоверный, то игнорируем его
        if not track.is_confirmed():
            continue

        # Полуаем ID трека для каждого объекта
        track_id = track.track_id
        class_id = track.get_det_class()
        ltrb = track.to_ltrb()

        xmin, ymin, xmax, ymax = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])

        # Отображаем координаты найденного объекта и ID найденного трека
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
        cv2.rectangle(frame, (xmin, ymin - 20), (xmin + 50, ymin), GREEN, -1)
        cv2.putText(frame, str(classes[class_id-1]), (xmin + 5, ymin - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)

    # Метка времени завершения обработки
    end = datetime.datetime.now()
    # Рассчитываем время, затраченное на обнаружение, и выводим этот показатель в консоль
    # print(f"Time to process 1 frame: {(end - start).total_seconds() * 1000:.0f} milliseconds")
    # calculate the frame per second and draw it on the frame
    fps = f"FPS: {1 / (end - start).total_seconds():.2f}"
    cv2.putText(frame, fps, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)
    # time.sleep(2)
    if results:
        for idx, c in enumerate(results):
            time_counter = time_counter + (end - start).total_seconds()
            if round(time_counter) % 5 == 0:
                Y = c[0][1]
                X = c[0][0]
                H = c[0][3]
                W = c[0][2]
                cv2.imwrite(f"C:\\Users\\User\\Desktop\\ML_CV\\DeepSort\\DeepSort\\{classes[c[-1]-1]}\\{counter}-{idx}.jpg", frame[Y:Y+H, X:X+W])
            counter += 1

    # Пересчитываем время в параметр FPS
    cv2.imshow("Frame", frame)
    writer.write(frame)
    if cv2.waitKey(1) == ord("q"):
        break

video_cap.release()
writer.release()
cv2.destroyAllWindows()