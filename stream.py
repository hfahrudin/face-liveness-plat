# face_match_webcam.py
import cv2
import face_recognition
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
import uvicorn
import numpy as np
from collections import deque

app = FastAPI()

# Load reference image and encode
reference_image = face_recognition.load_image_file("server/users/user3.jpg")
reference_encoding = face_recognition.face_encodings(reference_image)[0]

# Open webcam

camera = cv2.VideoCapture(0)
#TODO: LOGIC: MOVE TO CERTAIN DIRECTION UNTIL REACHING THRESHOLD, SUCCESS MEAN WHILE MOVING TOWARD DIRECTION X % OF TIME KEPT ITS IDENTITIY AND Y % MOVING TO THE RIGHT DIRECTION
#BUFFER FOR FACE MATCHER, IF 80% NOT TRUE< FAIL DIRECTLY
# ROI (x, y, w, h)
roi_x, roi_y, roi_w, roi_h = 275, 110, 140, 160  # adjust these values


MATCHER_BUFFER_SIZE = 30
MAX_VERTICAL_TH = 10
MAX_HORIZONTAL_TH = 10
matcher_buffer = []
movement_buffer = []

def get_head_direction(face_landmarks, 
                       horiz_threshold=0.05, 
                       up_threshold=0.12, 
                       down_threshold=0.55):
    """
    Estimate head direction from facial landmarks.
    
    Args:
        face_landmarks (dict): landmarks from face_recognition.face_landmarks
        horiz_threshold (int): pixel threshold for left/right decision
        up_threshold (float): relative nose position (0-1) above which head is 'UP'
        down_threshold (float): relative nose position (0-1) below which head is 'DOWN'
    
    Returns:
        str: "FORWARD", "LEFT", "RIGHT", "UP", "DOWN", or diagonal combos like "UP-LEFT".
    """
    nose = face_landmarks["nose_bridge"]
    left_eye = face_landmarks["left_eye"]
    right_eye = face_landmarks["right_eye"]
    chin = face_landmarks["chin"]

    # Nose tip
    nose_tip_x, nose_tip_y = nose[-1]

    # Eye centers
    left_eye_center = (sum([p[0] for p in left_eye]) / len(left_eye),
                       sum([p[1] for p in left_eye]) / len(left_eye))
    right_eye_center = (sum([p[0] for p in right_eye]) / len(right_eye),
                        sum([p[1] for p in right_eye]) / len(right_eye))

    eyes_center_x = (left_eye_center[0] + right_eye_center[0]) / 2
    eyes_center_y = (left_eye_center[1] + right_eye_center[1]) / 2

    # Chin center
    chin_center_x, chin_center_y = chin[len(chin)//2]

    # ---------- LEFT / RIGHT ----------
    face_width = right_eye_center[0] - left_eye_center[0]
    if nose_tip_x < eyes_center_x - face_width * horiz_threshold:
        horizontal = "LEFT"
    elif nose_tip_x > eyes_center_x + face_width *horiz_threshold:
        horizontal = "RIGHT"
    else:
        horizontal = "FORWARD"
        # ---------- UP / DOWN ----------
    face_height = chin_center_y - eyes_center_y
    nose_relative = (nose_tip_y - eyes_center_y) / face_height

    if nose_relative < up_threshold:
        vertical = "UP"
    elif nose_relative > down_threshold:
        vertical = "DOWN"
    else:
        vertical = "STRAIGHT"

    # ---------- FINAL RESULT ----------
    if vertical == "STRAIGHT":
        return horizontal  # LEFT, RIGHT, FORWARD
    elif horizontal == "FORWARD":
        return vertical    # UP, DOWN
    else:
        return horizontal
        # return f"{vertical}-{horizontal}"  # e.g. UP-LEFT
    
def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Draw ROI box on full frame
            cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 0, 0), 2)

            # Crop to ROI
            roi_frame = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

            # Resize ROI for faster detection
            small_frame = cv2.resize(roi_frame, (0, 0), fx=0.5, fy=0.5)

            # Detect face encodings and landmarks in ROI
            face_encodings = face_recognition.face_encodings(small_frame)
            face_landmarks_list = face_recognition.face_landmarks(small_frame)
            label = "No Face"
            if len(face_encodings) > 0:
                # Compare with reference encoding

                matches = face_recognition.compare_faces(reference_encoding, face_encodings, tolerance=0.4)
                if True in matches:
                    label = "Matched"
                    color = (0, 255, 0)  # green
                else:
                    label = "No Face"
                    color = (0, 0, 255)  # red

                # Draw landmarks
                for face_landmarks in face_landmarks_list:
                    label = get_head_direction(face_landmarks_list[0])
                    for feature, points in face_landmarks.items():
                        for (x, y) in points:
                            # Scale coords back to ROI size and shift
                            x = int(x * 2) + roi_x
                            y = int(y * 2) + roi_y
                            cv2.circle(frame, (x, y), 2, color, -1)

                # Draw label above ROI box
                cv2.putText(frame, label, (roi_x, roi_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.get("/")
def index():
    html_content = """
    <html>
        <head><title>Face Match Webcam</title></head>
        <body>
            <h1>Webcam Face Match</h1>
            <img src="/video_feed" width="640" height="480"/>
        </body>
    </html>
    """
    return Response(content=html_content, media_type="text/html")

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(gen_frames(),
                             media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
