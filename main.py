import cv2
import math


def task_1_image_processing():
    img = cv2.imread('variant-10.jpg')
    if img is None:
        print("Ошибка: Не удалось загрузить variant-10.jpg")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    
    cv2.imshow('Original Image', img)
    cv2.imshow('Threshold 150', thresh)
    
    print("Нажми любую клавишу или нажми на крестик окна, чтобы закрыть его...")
    
    while True:
        if cv2.waitKey(100) != -1:
            break
        if cv2.getWindowProperty('Original Image', cv2.WND_PROP_VISIBLE) < 1 or \
           cv2.getWindowProperty('Threshold 150', cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()


def task_2_video_tracking():
    cap = cv2.VideoCapture(0) 
    down_points = (640, 480)
    
    fly = cv2.imread('fly64.png', cv2.IMREAD_UNCHANGED)
    if fly is not None:
        fh, fw = fly.shape[:2]
        has_alpha = (fly.shape[2] == 4)
    else:
        print("Ошибка: Не удалось загрузить fly64.png")
        return

    sq_size = 150
    cx, cy = down_points[0] // 2, down_points[1] // 2
    sq_x1, sq_x2 = cx - sq_size // 2, cx + sq_size // 2
    sq_y1, sq_y2 = cy - sq_size // 2, cy + sq_size // 2

    is_flipped = False
    was_inside = False

    smooth_x, smooth_y, smooth_r = None, None, None
    alpha = 0.7  
    lost_frames = 0
    max_lost_frames = 15 

    print("Нажми 'q' или нажми на крестик окна, чтобы выйти...")
    
    cv2.namedWindow('Tracking & Fly', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Tracking & Fly', 960, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, down_points, interpolation=cv2.INTER_LINEAR)
        
        if is_flipped:
            frame = cv2.flip(frame, -1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        ret, thresh = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        cv2.rectangle(frame, (sq_x1, sq_y1), (sq_x2, sq_y2), (255, 0, 0), 2)
        
        valid_contours = []
        for c in contours:
            hull = cv2.convexHull(c)
            area = cv2.contourArea(hull)
            
            if 200 < area < 30000:
                perimeter = cv2.arcLength(hull, True)
                if perimeter == 0:
                    continue
                
                circularity = 4 * math.pi * (area / (perimeter * perimeter))
                
                if 0.82 <= circularity <= 1.2:
                    valid_contours.append(hull)

        if len(valid_contours) > 0:
            c = max(valid_contours, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(c)
            raw_x, raw_y, raw_r = int(x), int(y), int(radius)
            
            if smooth_x is None:
                smooth_x, smooth_y, smooth_r = raw_x, raw_y, raw_r
            else:
                smooth_x = int(alpha * raw_x + (1 - alpha) * smooth_x)
                smooth_y = int(alpha * raw_y + (1 - alpha) * smooth_y)
                smooth_r = int(alpha * raw_r + (1 - alpha) * smooth_r)
                
            lost_frames = 0 
        else:
            lost_frames += 1
            if lost_frames >= max_lost_frames:
                smooth_x, smooth_y, smooth_r = None, None, None

        if smooth_x is not None:
            a, b = smooth_x, smooth_y
            
            cv2.circle(frame, (smooth_x, smooth_y), smooth_r, (0, 255, 0), 2)
            cv2.circle(frame, (smooth_x, smooth_y), 3, (0, 0, 255), -1)
            
            is_inside = (sq_x1 <= a <= sq_x2 and sq_y1 <= b <= sq_y2)

            if is_inside and not was_inside:
                is_flipped = not is_flipped
            
            was_inside = is_inside

            y1 = int(b - fh // 2)
            y2 = int(y1 + fh)
            x1 = int(a - fw // 2)
            x2 = int(x1 + fw)

            y1_f = max(0, y1)
            y2_f = min(frame.shape[0], y2)
            x1_f = max(0, x1)
            x2_f = min(frame.shape[1], x2)

            if y1_f < y2_f and x1_f < x2_f:
                y1_fly = y1_f - y1
                y2_fly = y1_fly + (y2_f - y1_f)
                x1_fly = x1_f - x1
                x2_fly = x1_fly + (x2_f - x1_f)

                if has_alpha:
                    alpha_fly = fly[y1_fly:y2_fly, x1_fly:x2_fly, 3] / 255.0
                    alpha_frame = 1.0 - alpha_fly
                    
                    for c_idx in range(3):
                        color_fly = fly[y1_fly:y2_fly, x1_fly:x2_fly, c_idx]
                        color_frame = frame[y1_f:y2_f, x1_f:x2_f, c_idx]
                        frame[y1_f:y2_f, x1_f:x2_f, c_idx] = (alpha_fly * color_fly + alpha_frame * color_frame).astype('uint8')
                else:
                    frame[y1_f:y2_f, x1_f:x2_f] = fly[y1_fly:y2_fly, x1_fly:x2_fly, :3]
        else:
            was_inside = False

        cv2.imshow('Tracking & Fly', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.getWindowProperty('Tracking & Fly', cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    #task_1_image_processing()
    task_2_video_tracking()