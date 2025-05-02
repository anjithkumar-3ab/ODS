import torch
import cv2
from tkinter import Tk, filedialog, Button, Label
import os
import threading

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)

video_recording = False
video_writer = None
cap = None

def browse_from_device():
    """Let user select an image or video file."""
    Tk().withdraw()
    file_path = filedialog.askopenfilename(
        title="Select an image or video",
        filetypes=[("Media files", "*.jpg *.jpeg *.png *.mp4 *.avi *.mov *.mkv")]
    )
    return file_path

def capture_image():
    """Capture a single image from webcam."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return None

    print("Press SPACE to capture, ESC to cancel.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture.")
            break

        cv2.imshow("Capture Image - SPACE to capture", frame)
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            print("Cancelled.")
            break
        elif key == 32:  # SPACE
            path = "captured_image.jpg"
            cv2.imwrite(path, frame)
            print(f"Image saved to {path}")
            cap.release()
            cv2.destroyAllWindows()
            return path

    cap.release()
    cv2.destroyAllWindows()
    return None

def start_video_recording():
    global video_recording, video_writer, cap
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open webcam.")
        return

    width, height = int(cap.get(3)), int(cap.get(4))
    video_writer = cv2.VideoWriter("captured_video.avi", cv2.VideoWriter_fourcc(*'XVID'), 20.0, (width, height))
    video_recording = True
    print("Video recording started...")

    while video_recording:
        ret, frame = cap.read()
        if not ret:
            break
        video_writer.write(frame)
        cv2.imshow("Recording Video", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()

def stop_video_recording(root):
    global video_recording, cap, video_writer
    video_recording = False
    if cap:
        cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()
    print("Video recording stopped.")
    root.destroy()

def capture_video_gui():
    """Start a GUI to control video recording."""
    root = Tk()
    root.title("Camera Video Recorder")
    root.geometry("300x150")

    Label(root, text="Webcam Video Capture").pack(pady=10)

    Button(root, text="Start Video", command=lambda: threading.Thread(target=start_video_recording).start()).pack(pady=5)
    Button(root, text="Stop Video", command=lambda: stop_video_recording(root)).pack(pady=5)

    root.mainloop()
    return "captured_video.avi"

def process_image(path):
    """Run YOLOv5 on image."""
    results = model(path)
    results.print()
    results.show()
    results.save(save_dir='E:/image_classifier/runs/detect')
    print("Image results saved to E:/image_classifier/runs/detect")

def process_video(path):
    """Run YOLOv5 on video frame-by-frame."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("Could not open video.")
        return

    save_path = 'E:/image_classifier/runs/detect/video_output.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = None

    window_name = "YOLOv5 Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        rendered = results.render()[0]
        resized = cv2.resize(rendered, (800, 600))

        if out is None:
            height, width, _ = rendered.shape
            out = cv2.VideoWriter(save_path, fourcc, 20.0, (width, height))

        out.write(rendered)
        cv2.imshow(window_name, resized)

        if cv2.waitKey(1) & 0xFF == 27 or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            print("Stopped.")
            break

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    print(f"Video results saved to: {save_path}")

# --- Main Program ---

print("Choose input method:")
print("1. Browse from device")
print("2. Use camera")

choice = input("Enter 1 or 2: ").strip()
source = None
is_video = False

if choice == '1':
    source = browse_from_device()
    if source:
        ext = os.path.splitext(source)[-1].lower()
        is_video = ext in ['.mp4', '.avi', '.mov', '.mkv']
elif choice == '2':
    print("Choose capture type:")
    print("1. Capture image")
    print("2. Capture video")
    sub_choice = input("Enter 1 or 2: ").strip()
    if sub_choice == '1':
        source = capture_image()
        is_video = False
    elif sub_choice == '2':
        source = capture_video_gui()
        is_video = True
    else:
        print("Invalid sub-option.")
        exit()
else:
    print("Invalid choice.")
    exit()

if not source or not os.path.exists(source):
    print("No valid input received.")
    exit()

# Run YOLO inference
if is_video:
    process_video(source)
else:
    process_image(source)
