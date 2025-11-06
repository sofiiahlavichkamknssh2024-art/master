import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time
from collections import deque
from djitellopy import Tello

# Завантаження моделі
model = load_model('optimized_gesture_model.h5')

# Маппінг жестів
GESTURE_MAPPING = {
    0: "stop", 1: "up", 2: "down", 3: "forward", 5: "backward", 4: "left", 6: "right",
}

# Параметри виявлення руки
aweight = 0.5
ROI_TOP, ROI_BOTTOM = 225, 775
ROI_LEFT, ROI_RIGHT = 1080, 1580

# Параметри розпізнавання
CONFIDENCE_THRESHOLD = 0.7
MIN_CONTOUR_AREA = 8000
PREDICTION_INTERVAL = 3  # Частота виклику моделі

# таймер для посадки по жесту 'stop'
STOP_LAND_TIMER = 15.0 

# Швидкість руху дрона
DRONE_SPEED = 20 

# Налаштування адаптивного вікна
ADAPTIVE_WINDOW_MIN_PREDICTIONS = 3
ADAPTIVE_WINDOW_MAX_SECONDS = 0.8
ADAPTIVE_CONSENSUS_THRESHOLD = 0.7

# ГЛОБАЛЬНІ ЗМІННІ
tello = None
drone_connected = False
num_frames = 0
bg = None
prediction_history = deque(maxlen=30)
current_drone_command = None
last_command_sent = None
last_stable_confidence = 0.0
drone_state = "landed"
stop_gesture_start_time = None 

# ФУНКЦІЇ

def connect_to_drone():
    # Підʼєднання до дрона 
    global tello, drone_connected
    try:
        tello = Tello()
        tello.connect()
        print("Дрон успішно підключено!")
        drone_connected = True
        return True
    except Exception as e:
        print(f"Не вдалося підключитися до дрона: {e}")
        drone_connected = False
        return False

def run_avg(img, aweight):
    # Накопичує середнє значення фону для калібрування
    global bg
    if bg is None:
        bg = img.copy().astype('float')
        return
    cv2.accumulateWeighted(img, bg, aweight)

def segment(img, thres=25):
    #Сегментація
    global bg
    diff = cv2.absdiff(bg.astype('uint8'), img)
    _, thresholded = cv2.threshold(diff, thres, 255, cv2.THRESH_BINARY)
    thresholded = cv2.erode(thresholded, None, iterations=1)
    thresholded = cv2.dilate(thresholded, None, iterations=1)
    thresholded = cv2.medianBlur(thresholded, 5)
    contours, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    segmented = max(contours, key=cv2.contourArea)
    return (thresholded, segmented)

def preprocess_for_prediction(thresholded_img):
    # Бінарну маску для подачі на модель
    resized = cv2.resize(thresholded_img, (128, 128))
    normalized = resized / 255.0
    return np.expand_dims(normalized, axis=(0, -1))

def send_drone_command(command, confidence):
    global last_command_sent, drone_state, tello
    
    if not drone_connected or drone_state != "flying": 
        return

    # Надсилаємо тільки нову команду
    if command == last_command_sent:
        return

    print(f"Нова команда: {command} (впевненість: {confidence:.2f})")
    
    try:
        fwd_back, up_down, left_right, yaw = 0, 0, 0, 0

        if command == "up":
            up_down = DRONE_SPEED
        elif command == "down":
            up_down = -DRONE_SPEED
        elif command == "forward":
            fwd_back = DRONE_SPEED
        elif command == "backward":
            fwd_back = -DRONE_SPEED
        elif command == "left":
            left_right = DRONE_SPEED
        elif command == "right":
            left_right = -DRONE_SPEED
        elif command == "stop":
            pass 
        
        tello.send_rc_control(left_right, fwd_back, up_down, yaw)
        last_command_sent = command
        
    except Exception as e:
        print(f"Помилка при надсиланні команди: {e}")


def drone_takeoff():
    # Ініціація зльоту дрона
    global drone_state, tello, last_command_sent
    if drone_connected and drone_state == "landed":
        try:
            tello.takeoff()
            drone_state = "flying"
            last_command_sent = "takeoff"
            print("Дрон злітає...")
        except Exception as e:
            print(f"Помилка зльоту: {e}")

# Функція посадки чарез команду 'stop'
def drone_land():
    # Ініціює посадку дрона
    global drone_state, tello, last_command_sent, stop_gesture_start_time
    if drone_connected and drone_state == "flying":
        try:
            print("Команда на посадку...")
            tello.send_rc_control(0, 0, 0, 0) # Зупинка руху перед посадкою
            tello.land()
            drone_state = "landed"
            last_command_sent = "land"
            stop_gesture_start_time = None # Скидання таймеру
        except Exception as e:
            print(f"Помилка посадки: {e}")

def main():
    # Головний цикл програми
    global bg, num_frames, prediction_history, last_command_sent, current_drone_command, drone_state, last_stable_confidence, stop_gesture_start_time
    
    if not connect_to_drone():
        print("Дрон не підключено. Запуск в режимі симуляції.")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Помилка: Не вдалося відкрити камеру")
        return
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    
    real_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    real_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    prediction_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        clone = frame.copy()
        
        # Перевірка, що ROI не виходить за межі кадру
        safe_top = max(ROI_TOP, 0)
        safe_bottom = min(ROI_BOTTOM, int(real_height))
        safe_left = max(ROI_LEFT, 0)
        safe_right = min(ROI_RIGHT, int(real_width))

        roi = frame[safe_top:safe_bottom, safe_left:safe_right]
        
        # Помилка перевірки розміру ROI
        if roi.size == 0:
            print("Помилка ROI: Неправильні координати, ROI порожній.")
            continue
            
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 0)
        
        if num_frames < 30:
            run_avg(gray, aweight)
        else:
            hand = segment(gray)
            if hand is not None:
                thresholded, segmented = hand
                # Контур з урахуванням зсуву ROI
                cv2.drawContours(clone, [segmented + (safe_left, safe_top)], -1, (0, 0, 255), 2)
                cv2.imshow("Thresholded", thresholded)
                
                if cv2.contourArea(segmented) > MIN_CONTOUR_AREA:
                    prediction_counter += 1
                    if prediction_counter >= PREDICTION_INTERVAL:
                        processed = preprocess_for_prediction(thresholded)
                        predictions = model.predict(processed, verbose=0)
                        class_id = np.argmax(predictions[0])
                        confidence = predictions[0][class_id]
                        if confidence > CONFIDENCE_THRESHOLD and class_id in GESTURE_MAPPING:
                            prediction_history.append((GESTURE_MAPPING[class_id], confidence, time.time()))
                        prediction_counter = 0
                    
                    # Логіка адаптивного вікна
                    stable_command_found, stable_prediction, stable_confidence = False, None, 0.0
                    
                    if len(prediction_history) >= ADAPTIVE_WINDOW_MIN_PREDICTIONS:
                        recent_predictions = list(prediction_history)[-ADAPTIVE_WINDOW_MIN_PREDICTIONS:]
                        recent_gestures = [p[0] for p in recent_predictions]
                        if len(set(recent_gestures)) == 1:
                            stable_prediction = recent_gestures[0]
                            confidences = [p[1] for p in recent_predictions]
                            stable_confidence = sum(confidences) / len(confidences)
                            stable_command_found = True
                            prediction_history.clear()
                    
                    if not stable_command_found:
                        current_time = time.time()
                        valid_predictions = [p for p in prediction_history if current_time - p[2] <= ADAPTIVE_WINDOW_MAX_SECONDS]
                        if valid_predictions:
                            all_gestures = [p[0] for p in valid_predictions]
                            most_common = max(set(all_gestures), key=all_gestures.count)
                            if all_gestures.count(most_common) / len(all_gestures) >= ADAPTIVE_CONSENSUS_THRESHOLD:
                                stable_prediction = most_common
                                confidences = [p[1] for p in valid_predictions if p[0] == most_common]
                                stable_confidence = sum(confidences) / len(confidences)
                                stable_command_found = True
                    
                    if stable_command_found:
                        current_drone_command, last_stable_confidence = stable_prediction, stable_confidence
                        
                        # Логіка посадки за жестом "stop" > 15 секунд 
                        if stable_prediction == "stop":
                            if stop_gesture_start_time is None:
                                stop_gesture_start_time = time.time()
                                print(f"Жест 'stop' виявлено. Утримуйте {STOP_LAND_TIMER}с для посадки.")
                            else:
                                elapsed = time.time() - stop_gesture_start_time
                                if elapsed > STOP_LAND_TIMER and drone_state == "flying":
                                    print(f"Жест 'stop' утримується > {STOP_LAND_TIMER}с. Посадка...")
                                    drone_land() 
                                else:
                                    remaining = int(STOP_LAND_TIMER - elapsed) + 1
                                    cv2.putText(clone, f"LANDING IN: {remaining}s", (ROI_LEFT - 70, ROI_TOP + 25), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        else:
                            # Якщо жест змінився (не "stop"), скидання таймер
                            if stop_gesture_start_time is not None:
                                print("Таймер посадки скасовано.")
                            stop_gesture_start_time = None

                        if drone_state == "flying":
                            send_drone_command(stable_prediction, stable_confidence)
                else:
                    cv2.putText(clone, "BRING HAND CLOSER", (ROI_LEFT, ROI_TOP - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        #  ІНТЕРФЕЙС
        if num_frames < 30:
            cv2.putText(clone, "CALIBRATING BACKGROUND - REMOVE YOUR HAND", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            progress = int(num_frames / 30 * 100)
            cv2.rectangle(clone, (50, 70), (50 + progress * 3, 85), (0, 255, 0), -1)
        else:
            cv2.putText(clone, "READY", (ROI_RIGHT - 80, ROI_TOP - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if current_drone_command:
            text = f"Command: {current_drone_command} ({last_stable_confidence:.2f})"
            cv2.putText(clone, text, (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        state_text = f"State: {drone_state}"
        cv2.putText(clone, state_text, (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        connection_text = f"Drone: {'Connected' if drone_connected else 'Disconnected'}"
        cv2.putText(clone, connection_text, (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255) if drone_connected else (0, 0, 255), 2)
        
        # прямокутник ROI
        cv2.rectangle(clone, (safe_left, safe_top), (safe_right, safe_bottom), (0, 255, 0), 2)
        cv2.putText(clone, "DRONE GESTURE CONTROL", (50, 30), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 30, 255), 2)
        
        instructions = [
            "'t': Takeoff", "'l': Land", "'r': Recalibrate background", "'q': Exit",
            f"State: {drone_state}",
            "Drone: " + ("Connected" if drone_connected else "Disconnected"),
            "Current command: " + (current_drone_command if current_drone_command else "none")
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(clone, instruction, (50, 450 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        num_frames += 1
        
        cv2.imshow('Drone Gesture Control', clone)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            if drone_state == "flying": drone_land(); time.sleep(3)
            break
        elif key == ord('r'):
            bg, num_frames, prediction_history, last_command_sent, stop_gesture_start_time = None, 0, deque(maxlen=30), None, None
            print("Перекалібрування фону...")
        elif key == ord('t'):
            drone_takeoff()
        elif key == ord('l'):
            drone_land()
        elif key == ord('c'):
            connect_to_drone()
    
    if drone_connected:
        tello.end()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()