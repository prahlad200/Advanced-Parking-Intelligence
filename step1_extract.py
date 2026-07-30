import cv2

cap = cv2.VideoCapture("parking_test.mp4")

# Skip ahead to frame 100 (about 3 seconds in) to avoid initial black frames
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)

success, frame = cap.read()

if success:
    cv2.imwrite("parking_frame.png", frame)
    print("Success! Saved frame #100 as parking_frame.png.")
else:
    print("Error: Could not read frame 100. Try a lower frame number like 30.")

cap.release()