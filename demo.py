import cv2
import mediapipe as mp
import pandas as pd
from datetime import datetime
import time

# Initialize Mediapipe face detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Create or load the Excel file
excel_file = "face_count.xlsx"
try:
    data = pd.read_excel(excel_file)
except FileNotFoundError:
    # Create a new dataframe if the file does not exist
    data = pd.DataFrame(columns=["Date", "Time", "Face Count"])

# Function to update Excel
def update_excel(face_count):
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    new_entry = {"Date": date, "Time": time_str, "Face Count": face_count}
    data.loc[len(data)] = new_entry
    data.to_excel(excel_file, index=False)
    print(f"Updated Excel: {face_count} face(s) detected at {time_str} on {date}")

# Set the interval for updating the Excel file (6 seconds)
update_interval = 6  # seconds
last_update_time = time.time()

# Start webcam and process frames (use index 0 for the front camera)
cap = cv2.VideoCapture(1)
with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting...")
            break

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)

        # Count the number of faces
        face_count = len(results.detections) if results.detections else 0

        # Display the face count on the frame
        cv2.putText(frame, f"Faces: {face_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Face Detection", frame)

        # Update Excel file every 6 seconds
        if time.time() - last_update_time > update_interval:
            update_excel(face_count)
            last_update_time = time.time()

        # Break on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
