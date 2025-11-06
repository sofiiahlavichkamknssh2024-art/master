import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import json
import random
from tqdm import tqdm

# Шлях до датасету
DATASET_ROOT = "/Users/mac/Desktop/master/dataset"
# Кількість копій створениз для аугментації
AUGMENTATION_FACTOR = 2
# Розмір зображення
IMAGE_SIZE = (128, 128)

# Маппінг жестів
GESTURE_MAPPING = {
    0: "stop",
    1: "up",
    2: "down",
    3: "forward",
    5: "backward",
    4: "left",
    6: "right",
}
# Створення мапінгу класів
CLASS_NAMES = {command: idx for idx, command in GESTURE_MAPPING.items()}

def augment_image(img):
    # афінні перетворення
    rows, cols = img.shape[:2]
    
    # зсув (до 20%)
    max_shift = int(0.20 * min(rows, cols))
    tx = random.randint(-max_shift, max_shift)
    ty = random.randint(-max_shift, max_shift)
    M_shift = np.float32([[1, 0, tx], [0, 1, ty]])
    img = cv2.warpAffine(img, M_shift, (cols, rows), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    # масштабування (80-120%)
    scale = random.uniform(0.8, 1.2)
    M_scale = cv2.getRotationMatrix2D((cols/2, rows/2), 0, scale)
    img = cv2.warpAffine(img, M_scale, (cols, rows), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    # поворот (-25 до +25 градусів)
    angle = random.uniform(-25, 25)
    M_rot = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    img = cv2.warpAffine(img, M_rot, (cols, rows), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    return img

def load_and_augment_dataset(augmentation_factor=2):
    # Завантаження даних, зміна розміру, аугментація
    
    images = []
    labels = []
    
    print(f"Завантаження та аугментація даних з {DATASET_ROOT}...")
    
    for gesture_id_str in os.listdir(DATASET_ROOT):
        if not gesture_id_str.isdigit():
            continue
        
        gesture_id = int(gesture_id_str)
        gesture_path = os.path.join(DATASET_ROOT, gesture_id_str)
        
        if not os.path.isdir(gesture_path):
            continue
            
        img_files = [f for f in os.listdir(gesture_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_name in tqdm(img_files, desc=f"Обробка жесту {gesture_id}"):
            img_path = os.path.join(gesture_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                print(f"Помилка завантаження: {img_path}")
                continue
            # Зміна розміру зображення
            img_resized = cv2.resize(img, IMAGE_SIZE)

            # Додаємо оригінальне зображення
            images.append(img_resized)
            labels.append(gesture_id)
            
            # Додаємо аугментовані зображення
            for _ in range(augmentation_factor):
                augmented = augment_image(img_resized.copy())
                images.append(augmented)
                labels.append(gesture_id)

    # Перемішування даних
    combined = list(zip(images, labels))
    random.shuffle(combined)
    images, labels = zip(*combined)
    
    return np.array(images), np.array(labels)

def preprocess_dataset(X):
    # Застосовання нормалізації
    processed_images = []
    for img in tqdm(X, desc="Фінальна обробка"):
        normalized = img.astype(np.float32) / 255.0
        processed_images.append(normalized)
    return np.array(processed_images)

def main():
    os.makedirs("preprocessed", exist_ok=True)
    
    print("Початок обробки даних...")
    
    # Завантаження та розширення датасету
    X_augmented, y_augmented = load_and_augment_dataset(augmentation_factor=AUGMENTATION_FACTOR)
    
    if len(X_augmented) == 0:
        print("Помилка: не знайдено жодного зображення!")
        return
        
    print(f"\nЗагальний розмір розширеного датасету: {len(X_augmented)} зображень.")
    
    # Нормалізація
    print("\nНормалізація даних...")
    X_processed = preprocess_dataset(X_augmented)

    # Розподіл даних на тренувальний, валідаційний та тестовий набори
    print("\nРозділення датасету...")
    X_train, X_temp, y_train, y_temp = train_test_split(X_processed, y_augmented, test_size=0.30, random_state=42, stratify=y_augmented)
    
    X_val, X_test, y_val, y_test = train_test_split( X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)
    
    # Додавання розмірності каналу
    X_train_final = X_train[..., np.newaxis]
    X_val_final = X_val[..., np.newaxis]
    X_test_final = X_test[..., np.newaxis]
    
    print("\nЗбереження даних...")
    save_path = 'preprocessed'
    np.save(os.path.join(save_path, 'X_train.npy'), X_train_final)
    np.save(os.path.join(save_path, 'y_train.npy'), y_train)
    np.save(os.path.join(save_path, 'X_val.npy'), X_val_final)
    np.save(os.path.join(save_path, 'y_val.npy'), y_val)
    np.save(os.path.join(save_path, 'X_test.npy'), X_test_final)
    np.save(os.path.join(save_path, 'y_test.npy'), y_test)
    
    with open(os.path.join(save_path, 'label_mapping.json'), 'w') as f:
        json.dump({
            'gesture_mapping': GESTURE_MAPPING,
            'class_names': CLASS_NAMES
        }, f, indent=4)
    
    print("\nОбробку даних завершено!")
    print(f"Тренувальні дані: {len(X_train_final)} зразків")
    print(f"Валідаційні дані: {len(X_val_final)} зразків")
    print(f"Тестові дані: {len(X_test_final)} зразків")
    print(f"Кількість класів: {len(GESTURE_MAPPING)}")

if __name__ == "__main__":
    main()