import cv2
from ultralytics import YOLO

# Load your fine-tuned YOLO model
model = YOLO("results/yolov9t_ADAMW/weights/best.pt")  

# Open the default webcam (usually 0 for Mac)
cap = cv2.VideoCapture(0)

# Set frame size if needed
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height

print("[INFO] Press 'q' to quit...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame")
        break

    # Run detection
    results = model(frame)

    # Visualize predictions
    annotated_frame = results[0].plot()

    # Show the frame
    cv2.imshow("Crack Detection - Live", annotated_frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()