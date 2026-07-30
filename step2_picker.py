import cv2
import pickle

# Adjust these values based on how large the parking spots look in your video!
WIDTH, HEIGHT = 41, 20 

# Load the position list if it exists, otherwise create an empty list
try:
    with open('CarParkPos.pkl', 'rb') as f:
        posList = pickle.load(f)
except:
    posList = []

# Function to handle mouse clicks
def mouseClick(events, x, y, flags, params):
    # Left Mouse Button: Records the (x, y) coordinates and adds it to posList
    if events == cv2.EVENT_LBUTTONDOWN:
        posList.append((x, y))
        
    # Right Mouse Button: Loops through recorded positions and removes the spot if clicked inside
    if events == cv2.EVENT_RBUTTONDOWN:
        for i, pos in enumerate(posList):
            x1, y1 = pos
            if x1 < x < x1 + WIDTH and y1 < y < y1 + HEIGHT:
                posList.pop(i)
                break
                
    # Save the updated list into a binary file using pickle
    with open('CarParkPos.pkl', 'wb') as f:
        pickle.dump(posList, f)

# Create a window and attach the mouse callback function
cv2.namedWindow("Parking Lot Spot Selector")
cv2.setMouseCallback("Parking Lot Spot Selector", mouseClick)

while True:
    # Load the image in each iteration
    img = cv2.imread('parking_frame.png')
    
    if img is None:
        print("Error: 'parking_frame.png' not found.")
        break
        
    # Draw a magenta rectangle for each parking space position in posList
    for pos in posList:
        cv2.rectangle(img, pos, (pos[0] + WIDTH, pos[1] + HEIGHT), (255, 0, 255), 2)
        
    # Display instructions on the screen
    cv2.putText(
        img, 
        f"Spots: {len(posList)} | Left Click: Add | Right Click: Delete | 'q': Quit", 
        (20, 30), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.6, 
        (0, 255, 0), 
        2
    )

    # Display the updated image
    cv2.imshow("Parking Lot Spot Selector", img)
    
    # Refresh the image and wait for input
    key = cv2.waitKey(1)
    if key == ord('q') or key == 27: # Press 'q' or 'ESC' to close
        break

cv2.destroyAllWindows()