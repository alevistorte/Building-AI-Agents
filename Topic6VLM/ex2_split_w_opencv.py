import cv2
import os

# filename = "Empty_Room_Single_Person_s_Walk"
filename = "Animated_Empty_Room_Video_Generation"

script_dir = os.path.dirname(os.path.abspath(__file__))

cap = cv2.VideoCapture(os.path.join(script_dir, "videos", f"{filename}.mp4"))
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"Frames per second: {fps}")
interval = int(fps / 2)  # frame interval for ~0.5 seconds

frames = []
frame_num = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if frame_num % interval == 0:
        frames.append(frame)
    frame_num += 1

cap.release()

save_path = os.path.join(script_dir, "frames", filename)
os.makedirs(save_path, exist_ok=True)

for i, frame in enumerate(frames):
    cv2.imwrite(os.path.join(save_path, f"frame_{i:04d}.jpg"), frame)
