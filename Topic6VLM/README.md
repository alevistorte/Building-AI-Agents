# Topic 6: Vision Language Models (VLM)

## Overview

This topic explores multimodal agents that reason over images and video using LLaVA (Large Language and Vision Assistant) running locally via Ollama. Two exercises are covered: a multi-turn image chat agent exposed through both a CLI and a Gradio web interface (Exercise 1), and a frame-by-frame person detection pipeline applied to video (Exercise 2).

---

## Tasks

### Exercise 1 — Basic VLM Chat Agent

For this exercise, both a CLI and a WebApp were created. The chat was based on previous week project. Only LlaVa integration was added.

#### ex1_simple_agent.py — Multi-turn CLI Image Chat Agent

**File:** `simple_agent.py`
**Description:** A LangGraph-orchestrated CLI agent for multi-turn Q&A about a single image. The user supplies an image path at startup; the image is attached to the first user message as `images: [image_path]` and sent to LLaVA via `ollama.chat()`. Subsequent turns send only the accumulated text history so the model retains context without re-encoding the image. The graph structure is `START → get_user_input → call_llm → print_response → (loop)` with conditional routing for quit commands and verbose/quiet toggle.
**Key code snippet:**

```python
new_user_message: dict = {"role": "user", "content": user_input}
if not messages:
    new_user_message["images"] = [image_path]  # attach image on first turn only
messages.append(new_user_message)
response = ollama.chat(model="llava", messages=messages)
```

**Output:**

```
============================================================
Multi-turn Image Chat Agent  –  LLaVA via Ollama
============================================================

Enter the path to your image:
```

---

#### ex1_gradio_image_chat.py — Gradio Web Interface for Image Chat

**File:** `ex1_gradio_image_chat.py`
**Description:** Wraps the same LLaVA call in a Gradio web interface served at `http://localhost:7860`. Unlike the CLI agent, the LangGraph graph processes one turn at a time (`START → call_llm → END`); Gradio drives the conversation loop by passing accumulated `agent_messages` state back into each invocation. Uploading a new image resets conversation history.

**Key code snippet:**

```python
result = graph.invoke({
    "user_input":  user_message,
    "messages":    agent_messages,   # full history passed by Gradio state
    "image_path":  image_path,
})
updated_messages = result["messages"]
assistant_reply = updated_messages[-1]["content"]
```

**Screenshots are available in `./webapp_pic/`**
The app supports uploads as well as pictures taken with the webcam.

---

### Exercise 2 — Person Detection in Video Frames

#### testing_opencv.py — Frame Extraction Utility

**File:** `ex2_split_w_opencv.py`
**Description:** Uses OpenCV to extract frames from a local `.mp4` video file at a fixed rate of 0.5 seconds per frame (`interval = int(fps / 2)`). Frames are saved as sequential JPEG files (`frame_0000.jpg`, `frame_0001.jpg`, ...) into `frames/<filename>/`. This preprocessing step produces the input required by `ex2_person_detection.py`. Supports two video filenames by toggling the commented `filename` variable.
**Key code snippet:**

```python
cap = cv2.VideoCapture(os.path.join(script_dir, "videos", f"{filename}.mp4"))
fps = cap.get(cv2.CAP_PROP_FPS)
interval = int(fps / 2)   # one frame every ~0.5 seconds
...
cv2.imwrite(os.path.join(save_path, f"frame_{i:04d}.jpg"), frame)
```

**Output:**

```
Frames per second: 24.0
```

---

#### ex2_person_detection.py — Person Detection Using LLaVA on Video Frames

**File:** `person_detection.py`
**Description:** Iterates over all extracted frames in `frames/<filename>/` and queries LLaVA via `ollama.chat()` for each frame with the prompt "Is there a person visible in this image? Answer with only Yes or No." It tracks enter/exit events — recording a timestamp when detection changes from absent to present or vice versa — and writes a timeline log to `<filename>_log.txt`. This demonstrates applying a VLM as a zero-shot frame-level classifier within a video analysis pipeline.

**Key code snippet:**

```python
response = ollama.chat(
    model="llava",
    messages=[{
        "role": "user",
        "content": 'Is there a person visible in this image? Answer with only "Yes" or "No".',
        "images": [frame_path],
    }]
)
detected = response["message"]["content"].strip().lower().startswith("yes")

if detected and not person_present:
    events.append(("enter", timestamp))
    person_present = True
elif not detected and person_present:
    events.append(("exit", timestamp))
    person_present = False
```

**Output:**

See `Animated_Empty_Room_Video_Generation_log.txt` for the results while analyzing the video `Animated_Empty_Room_Video_Generation_log.mp4` and `Empty_Room_Single_Person_s_Walk_log.txt` for the results while analyzing the video `Empty_Room_Single_Person_s_Walk_log.mp4`

Videos are located in `videos/`

```
Analyzing frames from 'Empty_Room_Single_Person_s_Walk'...
--- Person presence timeline ---
Entered scene at 00:00.5  (0.5s)
Exited  scene at 00:07.0  (7.0s)

Analyzing frames from 'Animated_Empty_Room_Video_Generation'...
--- Person presence timeline ---
Entered scene at 00:00.0  (0.0s)
Exited  scene at 00:02.0  (2.0s)
```
