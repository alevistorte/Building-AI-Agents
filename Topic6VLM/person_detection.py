import ollama
import os

# filename = "Empty_Room_Single_Person_s_Walk"
filename = "Animated_Empty_Room_Video_Generation"
frame_interval_seconds = 0.5  # frames were sampled every 0.5s in testing_opencv.py

script_dir = os.path.dirname(os.path.abspath(__file__))
frames_dir = os.path.join(script_dir, "frames", filename)

frame_files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))

print(f"Analyzing {len(frame_files)} frames from '{filename}'...\n")

person_present = False
events = []
log_lines = []

for i, frame_file in enumerate(frame_files):
    frame_path = os.path.join(frames_dir, frame_file)
    timestamp = i * frame_interval_seconds

    response = ollama.chat(
        model="llava",
        messages=[{
            "role": "user",
            "content": 'Is there a person visible in this image? Answer with only "Yes" or "No".',
            "images": [frame_path],
        }]
    )

    answer = response["message"]["content"].strip().lower()
    detected = answer.startswith("yes")

    line = f"Frame {i:04d} | {timestamp:6.1f}s | {'Person' if detected else 'Empty '} | {answer}"
    print(line)
    log_lines.append(line)

    if detected and not person_present:
        events.append(("enter", timestamp))
        person_present = True
    elif not detected and person_present:
        events.append(("exit", timestamp))
        person_present = False

# If the person was still present at the last frame, record exit at the end
if person_present:
    last_timestamp = (len(frame_files) - 1) * frame_interval_seconds
    events.append(("exit", last_timestamp))

timeline_lines = ["\n--- Person presence timeline ---"]
if not events:
    timeline_lines.append("No person detected in any frame.")
else:
    for event, t in events:
        minutes = int(t // 60)
        seconds = t % 60
        label = "Entered" if event == "enter" else "Exited "
        timeline_lines.append(
            f"{label} scene at {minutes:02d}:{seconds:04.1f}  ({t:.1f}s)")

for line in timeline_lines:
    print(line)

log_path = os.path.join(script_dir, f"{filename}_log.txt")
with open(log_path, "w") as f:
    f.write(f"Analyzing {len(frame_files)} frames from '{filename}'...\n\n")
    f.write("\n".join(log_lines))
    f.write("\n".join(timeline_lines) + "\n")

print(f"\nLog saved to {log_path}")
