from fastapi import FastAPI, WebSocket, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import face_recognition
import cv2
import numpy as np
import base64
import json
import os
from fastapi.responses import JSONResponse
import shutil


SAVE_DIR = "frames"
os.makedirs(SAVE_DIR, exist_ok=True)


app = FastAPI()

# Allow all origins (for testing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load reference embeddings
reference_encodings = {}
for file in os.listdir("users"):
    if file.endswith(".jpg") or file.endswith(".png"):
        img = face_recognition.load_image_file(f"users/{file}")
        enc = face_recognition.face_encodings(img)[0]
        user_id = os.path.splitext(file)[0]
        reference_encodings[user_id] = enc

TOLERANCE = 0.4

# Placeholder liveness detection
def liveness_check(frame):
    return True


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
        horizontal = "RIGHT"
    elif nose_tip_x > eyes_center_x + face_width *horiz_threshold:
        horizontal = "LEFT"
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
        return horizontal, nose_tip_x # LEFT, RIGHT, FORWARD
    elif horizontal == "FORWARD":
        return vertical,  nose_relative   # UP, DOWN
    else:
        return vertical, nose_relative# e.g. UP-LEFT


@app.get("/users")
async def get_users():
    # Return the keys (user IDs) as a list of dicts for frontend dropdown
    return [{"id": user_id} for user_id in reference_encodings.keys()]


UPLOAD_DIR = "users"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/users/add")
async def add_user(image: UploadFile = File(...)):
    # Ensure it's an image
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type")

    # Extract username from file name (remove extension)
    username = os.path.splitext(image.filename)[0]

    # Save the image
    file_location = os.path.join(UPLOAD_DIR, image.filename)
    with open(file_location, "wb") as f:
        shutil.copyfileobj(image.file, f)
    img = face_recognition.load_image_file(file_location)
    enc = face_recognition.face_encodings(img)[0]
    user_id = os.path.splitext(file)[0]
    reference_encodings[user_id] = enc

    return JSONResponse({
        "username": username,
        "filename": image.filename,
        "path": file_location
    })
import glob

@app.delete("/users/delete/{username}")
async def delete_user(username: str):
    # Delete the face encoding
    if username in reference_encodings:
        del reference_encodings[username]
    else:
        raise HTTPException(status_code=404, detail="User not found in encodings")

    # Search for the image file with any common extension
    pattern = os.path.join(UPLOAD_DIR, f"{username}.*")  # match jpg, jpeg, png, etc.
    files = glob.glob(pattern)

    if not files:
        raise HTTPException(status_code=404, detail="User image file not found")

    # Delete all matching files (in case of duplicates)
    for file_path in files:
        os.remove(file_path)

    return JSONResponse({"detail": f"User '{username}' deleted successfully"})

def check_buffers(buffer, action_thresh=0.6, recog_thresh=0.6):
    action_buffer = buffer['action']
    recog_buffer = buffer['recog']

    # Ensure buffers are the same length
    if len(action_buffer) != len(recog_buffer) or len(action_buffer) == 0:
        return False

    # Calculate ratios
    action_ratio = sum(action_buffer) / len(action_buffer)
    recog_ratio = sum(recog_buffer) / len(recog_buffer)
    return action_ratio >= action_thresh and recog_ratio >= recog_thresh

# Challange:
# 1. Forward: just face forward for few sec
# 2. Left-up-right, need to move toward a threshold
from collections import deque

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    user_id = websocket.query_params.get("user_id")
    print(f"Authenticating user: {user_id}")
    challanges = ['forward','left', 'right', 'up']
    turn = 0   
    max_buffer = 20

    challange_buffer = {
        'recog' : deque(maxlen=max_buffer),
        'action' : deque(maxlen=max_buffer)
    }
    recog_buffer = challange_buffer['recog']
    action_buffer = challange_buffer['action']
    try:
        while True:
            if turn  > len(challanges):
                result = {"action": "authenticated", "state": "start"}
                await websocket.send_text(json.dumps(result))
                break
            message = await websocket.receive_text()
            data = json.loads(message)
            frame_data = base64.b64decode(data['frame'])
            np_arr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            ref = reference_encodings[user_id]
            
            # filename = os.path.join(SAVE_DIR, f"frame_{int(time.time()*1000)}.jpg")
            # cv2.imwrite(filename, frame)

            # await websocket.send_json(result)
            # if liveness_check(frame):
            #TODO: NEED TO MAKE SURE THAT ITS ONLY ONE PERSON
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_encodings = face_recognition.face_encodings(rgb_frame)
            face_landmarks_list = face_recognition.face_landmarks(rgb_frame)
            for i, (encoding, landmarks) in enumerate(zip(face_encodings, face_landmarks_list)):
                matches = face_recognition.compare_faces(
                    [ref], encoding, tolerance=TOLERANCE
                )
                label, debug = get_head_direction(landmarks)

                if True in matches:
                    recog_buffer.append(True)
                else:
                    recog_buffer.append(False)
                
                if label.lower() == challanges[turn].lower():
                    action_buffer.append(True)
                else:
                    action_buffer.append(False)
                
                break
      
            if len(recog_buffer) > max_buffer-1:
                passed = check_buffers(challange_buffer)
                if passed:
                    print(f'{challanges[turn]} Challange Passed')
                    turn+=1
                else:
                    print("fail")
                recog_buffer.clear()
                action_buffer.clear()

            if turn  >= len(challanges):
                result = {"action": "authenticated", "state": "start"}
            else:
                result = {"action": challanges[turn], "state": "start"}

            await websocket.send_text(json.dumps(result))

    except Exception as e:
        print("Client disconnected:", e)
