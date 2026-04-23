import cv2
import math

# ==========================================
# КОНСТАНТЫ ДЛЯ ЗАДАЧИ 1
# ==========================================
IMAGE_FILENAME = 'variant-10.jpg'
THRESHOLD_VALUE_TASK1 = 150
MAX_COLOR_VALUE = 255

# ==========================================
# КОНСТАНТЫ ДЛЯ ЗАДАЧИ 2
# ==========================================
FLY_FILENAME = 'fly64.png'
CAMERA_INDEX = 0
FRAME_RESOLUTION = (640, 480)
SQUARE_SIZE = 150

# Настройки компьютерного зрения (CV)
BLUR_KERNEL_SIZE = (21, 21)
MORPH_KERNEL_SIZE = (9, 9)
THRESHOLD_VALUE_TASK2 = 110

# Фильтры поиска контуров
MIN_CONTOUR_AREA = 200
MAX_CONTOUR_AREA = 30000
MIN_CIRCULARITY = 0.82
MAX_CIRCULARITY = 1.2

# Настройки сглаживания движений и трекинга
SMOOTHING_ALPHA = 0.7
MAX_LOST_FRAMES = 15


def task_1_image_processing() -> None:
    """
    Загружает изображение, переводит его в градации серого и применяет пороговую бинаризацию.
    Отображает исходное и обработанное изображения в отдельных окнах.
    """
    img = cv2.imread(IMAGE_FILENAME)
    if img is None:
        print(f"Ошибка: Не удалось загрузить {IMAGE_FILENAME}")
        return
    
    # Перевод в оттенки серого и бинаризация
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, THRESHOLD_VALUE_TASK1, MAX_COLOR_VALUE, cv2.THRESH_BINARY)
    
    cv2.imshow('Original Image', img)
    cv2.imshow(f'Threshold {THRESHOLD_VALUE_TASK1}', thresh)
    
    print("Нажми любую клавишу или нажми на крестик окна, чтобы закрыть его...")
    
    # Ожидание закрытия окна пользователем
    while True:
        if cv2.waitKey(100) != -1:
            break
        if cv2.getWindowProperty('Original Image', cv2.WND_PROP_VISIBLE) < 1 or \
           cv2.getWindowProperty(f'Threshold {THRESHOLD_VALUE_TASK1}', cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()


def task_2_video_tracking() -> None:
    """
    Захватывает видео с веб-камеры, отслеживает круглый объект и накладывает 
    изображение-спрайт поверх найденного объекта. Если объект попадает 
    в центральный квадрат, видео отзеркаливается.
    """
    cap = cv2.VideoCapture(CAMERA_INDEX) 
    
    # Загрузка спрайта (мухи)
    fly = cv2.imread(FLY_FILENAME, cv2.IMREAD_UNCHANGED)
    if fly is not None:
        fh, fw = fly.shape[:2]
        has_alpha = (fly.shape[2] == 4)
    else:
        print(f"Ошибка: Не удалось загрузить {FLY_FILENAME}")
        cap.release()
        return

    # Координаты центральной зоны (квадрата)
    cx, cy = FRAME_RESOLUTION[0] // 2, FRAME_RESOLUTION[1] // 2
    sq_x1, sq_x2 = cx - SQUARE_SIZE // 2, cx + SQUARE_SIZE // 2
    sq_y1, sq_y2 = cy - SQUARE_SIZE // 2, cy + SQUARE_SIZE // 2

    # Состояния
    is_flipped = False
    was_inside = False

    # Переменные для экспоненциального сглаживания координат
    smooth_x, smooth_y, smooth_r = None, None, None
    lost_frames = 0

    print("Нажми 'q' или нажми на крестик окна, чтобы выйти...")
    
    cv2.namedWindow('Tracking & Fly', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Tracking & Fly', 960, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Ошибка: Не удалось получить кадр с камеры.")
            break

        # Масштабирование и возможное отзеркаливание кадра
        frame = cv2.resize(frame, FRAME_RESOLUTION, interpolation=cv2.INTER_LINEAR)
        if is_flipped:
            frame = cv2.flip(frame, -1)

        # --- 1. Предобработка кадра ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, BLUR_KERNEL_SIZE, 0)
        
        # Бинаризация с использованием метода Оцу и инверсией
        _, thresh = cv2.threshold(gray, THRESHOLD_VALUE_TASK2, MAX_COLOR_VALUE, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Морфологическое замыкание для удаления мелких "дырок" внутри объектов
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL_SIZE)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Поиск контуров
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # Отрисовка центрального квадрата (триггерной зоны)
        cv2.rectangle(frame, (sq_x1, sq_y1), (sq_x2, sq_y2), (255, 0, 0), 2)
        
        # --- 2. Фильтрация контуров ---
        valid_contours = []
        for c in contours:
            hull = cv2.convexHull(c)
            area = cv2.contourArea(hull)
            
            # Проверка по площади
            if MIN_CONTOUR_AREA < area < MAX_CONTOUR_AREA:
                perimeter = cv2.arcLength(hull, True)
                if perimeter == 0:
                    continue
                
                # Проверка на округлость
                circularity = 4 * math.pi * (area / (perimeter * perimeter))
                if MIN_CIRCULARITY <= circularity <= MAX_CIRCULARITY:
                    valid_contours.append(hull)

        # --- 3. Трекинг и сглаживание движения ---
        if valid_contours:
            # Выбираем самый крупный контур из прошедших проверку
            c = max(valid_contours, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(c)
            raw_x, raw_y, raw_r = int(x), int(y), int(radius)
            
            # Сглаживание рывков (Exponential Moving Average)
            if smooth_x is None:
                smooth_x, smooth_y, smooth_r = raw_x, raw_y, raw_r
            else:
                smooth_x = int(SMOOTHING_ALPHA * raw_x + (1 - SMOOTHING_ALPHA) * smooth_x)
                smooth_y = int(SMOOTHING_ALPHA * raw_y + (1 - SMOOTHING_ALPHA) * smooth_y)
                smooth_r = int(SMOOTHING_ALPHA * raw_r + (1 - SMOOTHING_ALPHA) * smooth_r)
                
            lost_frames = 0 
        else:
            # Если объект потерян, ждем несколько кадров перед сбросом
            lost_frames += 1
            if lost_frames >= MAX_LOST_FRAMES:
                smooth_x, smooth_y, smooth_r = None, None, None

        # --- 4. Отрисовка маркеров и логика взаимодействия ---
        if smooth_x is not None:
            a, b = smooth_x, smooth_y
            
            # Маркер объекта
            cv2.circle(frame, (a, b), smooth_r, (0, 255, 0), 2)
            cv2.circle(frame, (a, b), 3, (0, 0, 255), -1)
            
            # Проверка: находится ли объект внутри центрального квадрата?
            is_inside = (sq_x1 <= a <= sq_x2 and sq_y1 <= b <= sq_y2)

            # Меняем отзеркаливание только в момент ВХОДА в зону
            if is_inside and not was_inside:
                is_flipped = not is_flipped
            
            was_inside = is_inside

            # --- 5. Наложение спрайта (мухи) ---
            # Вычисляем границы для вставки спрайта
            y1 = int(b - fh // 2)
            y2 = int(y1 + fh)
            x1 = int(a - fw // 2)
            x2 = int(x1 + fw)

            # Обрезаем координаты, чтобы они не выходили за пределы экрана
            y1_f, y2_f = max(0, y1), min(frame.shape[0], y2)
            x1_f, x2_f = max(0, x1), min(frame.shape[1], x2)

            if y1_f < y2_f and x1_f < x2_f:
                # Координаты внутри самого спрайта (если он частично ушел за край экрана)
                y1_fly, y2_fly = y1_f - y1, (y1_f - y1) + (y2_f - y1_f)
                x1_fly, x2_fly = x1_f - x1, (x1_f - x1) + (x2_f - x1_f)

                if has_alpha:
                    # Наложение с учетом прозрачности (Альфа-канал)
                    alpha_fly = fly[y1_fly:y2_fly, x1_fly:x2_fly, 3] / 255.0
                    alpha_frame = 1.0 - alpha_fly
                    
                    for c_idx in range(3):
                        color_fly = fly[y1_fly:y2_fly, x1_fly:x2_fly, c_idx]
                        color_frame = frame[y1_f:y2_f, x1_f:x2_f, c_idx]
                        frame[y1_f:y2_f, x1_f:x2_f, c_idx] = (alpha_fly * color_fly + alpha_frame * color_frame).astype('uint8')
                else:
                    # Прямое копирование пикселей, если прозрачности нет
                    frame[y1_f:y2_f, x1_f:x2_f] = fly[y1_fly:y2_fly, x1_fly:x2_fly, :3]
        else:
            was_inside = False

        cv2.imshow('Tracking & Fly', frame)
        
        # Обработка выхода из программы
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.getWindowProperty('Tracking & Fly', cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # Раскомментируй нужную задачу для запуска:
    
    # task_1_image_processing()
    task_2_video_tracking()