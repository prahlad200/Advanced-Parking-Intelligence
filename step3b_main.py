import cv2
import pickle
import numpy as np
import sqlite3
import time
from datetime import datetime
import json

WIDTH, HEIGHT = 41, 20 
VIDEO_PATH = 'parking_test.mp4'

# --- 1. Database Setup (Added Violations Table) ---
conn = sqlite3.connect('parking_data.db')
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS occupancy (timestamp DATETIME, available_spots INTEGER, total_spots INTEGER)''')
cursor.execute('''CREATE TABLE IF NOT EXISTS violations (timestamp DATETIME, spot_id INTEGER)''')
conn.commit()

try:
    with open('CarParkPos.pkl', 'rb') as f:
        posList = pickle.load(f)
except:
    posList = []

# --- 2. State Management for Overstay Detection ---
# Dictionary to track {spot_index: {"state": "empty"/"occupied", "since": timestamp, "violation_logged": False}}
spot_memory = {i: {"state": "empty", "since": time.time(), "violation_logged": False} for i in range(len(posList))}
OVERSTAY_LIMIT = 15 # 15 seconds for live presentation demo

def check_parking_space(img_processed, img_display):
    free_spots = 0
    
    # We use enumerate(posList) so we know EXACTLY which spot we are looking at (Spot 0, Spot 1, etc.)
    for i, pos in enumerate(posList):
        x, y = pos
        img_crop = img_processed[y:y+HEIGHT, x:x+WIDTH]
        count = cv2.countNonZero(img_crop)
        
        if count < 250:
            color = (0, 255, 0) # Green / Empty
            thickness = 2
            free_spots += 1
            # Reset memory when car leaves
            spot_memory[i]["state"] = "empty"
            spot_memory[i]["since"] = time.time()
            spot_memory[i]["violation_logged"] = False
        else:
            # Car is present
            if spot_memory[i]["state"] == "empty":
                spot_memory[i]["state"] = "occupied"
                spot_memory[i]["since"] = time.time()
                
            # Calculate how long it's been there
            time_parked = time.time() - spot_memory[i]["since"]
            
            if time_parked > OVERSTAY_LIMIT:
                color = (0, 165, 255) # ORANGE in BGR: Parking Violation!
                thickness = 3
                
                # Log violation to database only ONCE
                if not spot_memory[i]["violation_logged"]:
                    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    cursor.execute("INSERT INTO violations (timestamp, spot_id) VALUES (?, ?)", (now_str, i))
                    conn.commit()
                    spot_memory[i]["violation_logged"] = True
            else:
                color = (0, 0, 255) # Red / Normally Occupied
                thickness = 2
                
        cv2.rectangle(img_display, pos, (pos[0] + WIDTH, pos[1] + HEIGHT), color, thickness)

    return free_spots

# --- 3. Start Video Loop ---
cap = cv2.VideoCapture(VIDEO_PATH)
last_log_time = time.time()

while True:
    success, frame = cap.read()
    if not success:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 1)
    img_thresh = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)
    img_median = cv2.medianBlur(img_thresh, 5)
    kernel = np.ones((3, 3), np.uint8)
    img_dilate = cv2.dilate(img_median, kernel, iterations=1)

    free_spots = check_parking_space(img_dilate, frame)

    # Database Logging (Once per second)
    current_time = time.time()
    if current_time - last_log_time >= 1.0:
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("INSERT INTO occupancy (timestamp, available_spots, total_spots) VALUES (?, ?, ?)", 
                       (now_str, free_spots, len(posList)))
        conn.commit()
        last_log_time = current_time
        
        cv2.imwrite("live_frame.jpg", frame)
        
        # NEW: Dump individual spot states for the Digital Twin Map
        with open("spot_states.json", "w") as f:
            json.dump(spot_memory, f)

    # UI
    cv2.rectangle(frame, (20, 20), (500, 80), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, f"Available Spots: {free_spots}/{len(posList)}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

    cv2.imshow("Automated Parking Lot Monitor", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
conn.close()