import cv2
import asyncio
import websockets
import base64
import json

async def stream_video():
    uri = "ws://localhost:8000/ws"
    async with websockets.connect(uri) as websocket:
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')

            # Send frame to server
            await websocket.send(json.dumps({"frame": jpg_as_text}))

            # Receive server response
            response = await websocket.recv()
            result = json.loads(response)
            if result["authenticated"]:
                print(f"ACCESS GRANTED: {result['user_id']}")
            else:
                print("ACCESS DENIED")

            cv2.imshow("Client Webcam", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

asyncio.get_event_loop().run_until_complete(stream_video())
