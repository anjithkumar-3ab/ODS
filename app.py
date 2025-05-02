import torch
import cv2
from tkinter import Tk, filedialog, Button, Label
from PIL import Image, ImageTk
import os
import threading

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)

# Global for camera recording
recording = False
cap = None
out = None


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


def capture_video_gui():
    """Combined GUI for webcam preview and video recording."""
    def show_frame():
        global recording, cap, out
        if not cap.isOpened():
            return

        ret, frame = cap.read()
        if ret:
            # Display in GUI
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(cv2image, (400, 300))
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            video_label.imgtk = imgtk
            video_label.configure(image=imgtk)

            if recording:
                out.write(frame)

        video_label.after(10, show_frame)

    def start_recording():
        global recording
        recording = True
        print("Recording started...")

    def stop_recording():
        global recording, cap, out
        recording = False
        print("Recording stopped.")
        cap.release()
        if out:
            out.release()
        root.destroy()

    # Setup GUI
    root = Tk()
    root.title("Video Recorder")
    root.geometry("450x400")

    video_label = Label(root)
    video_label.pack()

    Button(root, text="Start Recording", command=start_recording).pack(pady=5)
    Button(root, text="Stop and Save", command=stop_recording).pack(pady=5)

    # Initialize webcam
    global cap, out
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open webcam.")
        root.destroy()
        return None

    width, height = int(cap.get(3)), int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter("captured_video.avi", fourcc, 20.0, (width, height))

    show_frame()
    root.mainloop()
    return "captured_video.avi"


def process_image(path):
    """Run YOLOv5 on an image."""
    results = model(path)
    results.print()
    results.show()
    results.save(save_dir='E:/image_classifier/runs/detect')
    print("Image results saved to E:/image_classifier/runs/detect")


def process_video(path):
    """Run YOLOv5 on a video file."""
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
            break

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    print(f"Video results saved to: {save_path}")


# --- Main Program ---
if __name__ == '__main__':
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
