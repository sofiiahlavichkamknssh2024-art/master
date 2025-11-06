import cv2
import numpy as np
import os
import time
from djitellopy import Tello

# Ініціалізація дрону Tello
tello = Tello()
tello.connect()
tello.streamon()

print(f"Battery: {tello.get_battery()}%")

# Параметри для виявлення руки
aweight = 0.5
num_frames = 0
bg = None

# Налаштування області інтересу (ROI)
ROI_TOP, ROI_BOTTOM = 100, 400
ROI_LEFT, ROI_RIGHT = 200, 500  

# Параметри запису датасету
RECORDING = False
CURRENT_GESTURE = None
SAVE_INTERVAL = 5  # Зберігати кожні 5 кадрів
MIN_CONTOUR_AREA_RECORDING = 3000  # Мінімальна площа контуру для запису
save_count = 0
last_save_time = 0

def run_avg(img, aweight):
    global bg
    if bg is None:
        bg = img.copy().astype('float')
        return
    cv2.accumulateWeighted(img, bg, aweight)

def segment(img, thres=25):
    global bg
    diff = cv2.absdiff(bg.astype('uint8'), img)
    _, thresholded = cv2.threshold(diff, thres, 255, cv2.THRESH_BINARY)
    
    thresholded = cv2.erode(thresholded, None, iterations=1)
    thresholded = cv2.dilate(thresholded, None, iterations=2)
    thresholded = cv2.medianBlur(thresholded, 5)
    
    contours, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    segmented = max(contours, key=cv2.contourArea)
    return (thresholded, segmented)

def save_thresholded_image(gesture_id, image):
    global save_count, last_save_time
    timestamp = int(time.time() * 1000)
    
    # Створення папки для жесту
    gesture_dir = f"dataset/{gesture_id}"
    os.makedirs(gesture_dir, exist_ok=True)
    
    # Збереження зображення
    filename = os.path.join(gesture_dir, f"{gesture_id}_{timestamp}.png")
    cv2.imwrite(filename, image)
    save_count += 1
    last_save_time = time.time()
    return filename

def main():
    global bg, num_frames, RECORDING, CURRENT_GESTURE, save_count, last_save_time
    
    frame_counter = 0
    
    print("Dataset Collection Instructions:")
    print("1. Keep background static during calibration")
    print("2. Place hand inside the green rectangle")
    print("3. Perform gestures facing the drone")
    print("4. Press 'r' to recalibrate background")
    print("5. Press 'q' to quit")
    print("\nRecording Controls:")
    print("6. Press 0-6 to start recording for that gesture")
    print("7. Press 'x' to stop recording")
    
    # Визначення доступних жестів для відображення
    AVAILABLE_GESTURES = {
        0: "stop",
        1: "up",
        2: "down",
        3: "forward",
        5: "backward",
        4: "left",
        6: "right",
    }
    
    while True:
        # Отримання кадру від Tello
        frame = tello.get_frame_read().frame
        if frame is None or frame.size == 0:
            print("Error: Can't read frame from drone")
            time.sleep(0.1)
            continue
        
        frame = cv2.resize(frame, (1000, 750))
        
        # Відзеркалення зображення
        frame = cv2.flip(frame, 1)
        
        clone = frame.copy()
        
        roi = frame[ROI_TOP:ROI_BOTTOM, ROI_LEFT:ROI_RIGHT]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 0)
        
        if num_frames < 30:
            run_avg(gray, aweight)
            cv2.putText(clone, "BACKGROUND CALIBRATION - REMOVE HAND", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            progress = int(num_frames / 30 * 100)
            cv2.rectangle(clone, (50, 70), (50 + progress * 3, 85), (0, 255, 0), -1)
        else:
            hand = segment(gray)
            
            if hand is not None:
                thresholded, segmented = hand
                contour_area = cv2.contourArea(segmented)
                
                cv2.drawContours(clone, [segmented + (ROI_LEFT, ROI_TOP)], -1, (0, 0, 255), 2)
                cv2.imshow("Thresholded", thresholded)
                
                # Запис датасету
                if RECORDING and CURRENT_GESTURE is not None:
                    frame_counter += 1
                    
                    # Перевірка часу та умов для збереження
                    current_time = time.time()
                    save_ready = (
                        frame_counter % SAVE_INTERVAL == 0 and 
                        contour_area > MIN_CONTOUR_AREA_RECORDING and
                        current_time - last_save_time > 0.1  # Не частіше 10 кадрів на секунду
                    )
                    
                    if save_ready:
                        filename = save_thresholded_image(CURRENT_GESTURE, thresholded)
                        print(f"Saved: {filename}")
                else:
                    
                    cv2.putText(clone, "Press 0-6 to start recording", (50, 150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Відображення статусу запису
        if RECORDING and CURRENT_GESTURE is not None:
            gesture_name = AVAILABLE_GESTURES.get(CURRENT_GESTURE, f"Gesture {CURRENT_GESTURE}")
            status_text = f"Recording: {gesture_name} [{save_count}]"
            cv2.putText(clone, status_text, (50, 110), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        elif CURRENT_GESTURE is not None:
            gesture_name = AVAILABLE_GESTURES.get(CURRENT_GESTURE, f"Gesture {CURRENT_GESTURE}")
            cv2.putText(clone, f"Ready: {gesture_name}", (50, 110), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.rectangle(clone, (ROI_LEFT, ROI_TOP), (ROI_RIGHT, ROI_BOTTOM), (0, 255, 0), 2)
        cv2.putText(clone, f"Tello Dataset Collection | Battery: {tello.get_battery()}%", (50, 30), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 30, 255), 2)
        cv2.putText(clone, "'q': exit | 'r': recalibrate | 0-6: record gesture | 'x': stop", (50, 700), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if num_frames >= 30:
            cv2.putText(clone, "READY", (ROI_RIGHT - 70, ROI_TOP - 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        num_frames += 1
        
        cv2.imshow('Tello Dataset Collection', clone)
        
        # Обробка клавіш
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            bg = None
            num_frames = 0
            print("Recalibrating background...")
        elif key == ord('x'):
            RECORDING = False
            print(f"Stopped recording. Total saved: {save_count}")
        elif key >= 48 and key <= 54:  # 0-6
            gesture_num = key - 48
            CURRENT_GESTURE = gesture_num
            RECORDING = True
            save_count = 0
            frame_counter = 0
            gesture_name = AVAILABLE_GESTURES.get(gesture_num, f"Gesture {gesture_num}")
            print(f"Started recording for: {gesture_name}")
    
    # Завершення роботи
    tello.streamoff()
    tello.end()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Створення папки для збереження датасету
    os.makedirs("dataset", exist_ok=True)
    main()