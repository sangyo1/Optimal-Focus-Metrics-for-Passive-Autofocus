import cv2
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw, ImageFont
from focus_monitor import FocusMonitor
import sys
import numpy as np

class DemoApp:
    def __init__(self, root, camera_index=0):
        self.root = root
        self.root.title("Focus Monitor Demo")

        # Initialize webcam with the specified camera index
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise ValueError(f"Unable to access camera at index {camera_index}")
        
        # Set desired frame dimensions
        self.width, self.height = 640, 480
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # Metrics mapping: (Display Name, Internal Name)
        self.metrics = [
            ('Variance of Sobel', 'sobel'),
            ('Squared Gradient', 'squared_gradient'),
            ('Squared Sobel', 'squared_sobel'),
            ('FSWM', 'fswm'),
            ('FFT', 'fft'),
            ('Mix Sobel', 'mix_sobel'),
            ('Sobel+Laplacian', 'sobel_laplacian'),
            ('Sobel+FSWM', 'combined_focus_measure'),
            ('Sobel+FFT', 'combined_focus_measure2'),
        ]

        # Initialize a FocusMonitor instance for each metric
        self.focus_monitors = {}
        for display_name, internal_name in self.metrics:
            fm = FocusMonitor()
            fm.set_metric(display_name)
            self.focus_monitors[display_name] = fm

        # Create UI elements
        self.create_widgets()

        # Start the video feed loop
        self.update_frame()

    def create_widgets(self):
        # Canvas for displaying the original image
        self.canvas_original = tk.Canvas(self.root, width=self.width, height=self.height)
        self.canvas_original.grid(row=0, column=0, rowspan=2)

        # Create canvases for each focus metric
        self.metric_canvases = {}
        self.metric_images = {}
        num_metrics = len(self.metrics)
        num_columns = 5
        num_rows = (num_metrics + num_columns - 1) // num_columns

        # Define the size for each metric canvas
        metric_canvas_width = self.width // 5
        metric_canvas_height = self.height // 2

        # Create metric canvases in a grid layout
        for idx, (display_name, _) in enumerate(self.metrics):
            row = idx // num_columns
            col = idx % num_columns + 1 
            canvas = tk.Canvas(self.root, width=metric_canvas_width, height=metric_canvas_height)
            canvas.grid(row=row, column=col)
            self.metric_canvases[display_name] = canvas

        # Adjust the window size
        total_width = self.width + num_columns * metric_canvas_width
        total_height = max(self.height, num_rows * metric_canvas_height)
        self.root.geometry(f"{total_width}x{total_height}")

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, (self.width, self.height))

            try:
                original_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                self.update_canvas(self.canvas_original, original_image)

                for display_name, fm in self.focus_monitors.items():
                    focus_value, focus_image = fm.measure_focus(frame)

                    if focus_image is not None and focus_image.size > 0:
                        if len(focus_image.shape) == 2:
                            focus_image = cv2.cvtColor(focus_image, cv2.COLOR_GRAY2RGB)
                        elif focus_image.shape[2] == 1:
                            focus_image = cv2.cvtColor(focus_image, cv2.COLOR_GRAY2RGB)

                        focus_display_image = np.zeros_like(frame)
                        height, width, _ = frame.shape
                        roi_height, roi_width, _ = focus_image.shape
                        x0 = int((width - roi_width) / 2)
                        y0 = int((height - roi_height) / 2)
                        focus_display_image[y0:y0+roi_height, x0:x0+roi_width] = focus_image

                        focus_display_image = cv2.resize(focus_display_image, (self.width // 5, self.height // 2))

                        metric_image = Image.fromarray(cv2.cvtColor(focus_display_image, cv2.COLOR_BGR2RGB))

                        draw = ImageDraw.Draw(metric_image)
                        try:
                            font = ImageFont.truetype("arial.ttf", 12)
                        except IOError:
                            font = ImageFont.load_default()
                        text = f"{display_name}\nFocus Value: {focus_value:.2f}"
                        draw.text((10, 10), text, fill=(255, 255, 255), font=font)

                        # Update the canvas
                        self.update_canvas(self.metric_canvases[display_name], metric_image)
                    else:
                        blank_image = Image.new("RGB", (self.width // 5, self.height // 2))

                        draw = ImageDraw.Draw(blank_image)
                        try:
                            font = ImageFont.truetype("arial.ttf", 12)
                        except IOError:
                            font = ImageFont.load_default()
                        text = f"{display_name}\nFocus Value: N/A"
                        draw.text((10, 10), text, fill=(255, 255, 255), font=font)

                        # Update the canvas
                        self.update_canvas(self.metric_canvases[display_name], blank_image)
            except Exception as e:
                print(f"Error processing frame: {e}")
        else:
            print("Failed to read frame from camera.")

        # Schedule the next frame update
        self.root.after(10, self.update_frame)

    def update_canvas(self, canvas, image):
        # Convert PIL image to Tkinter-compatible image
        tk_image = ImageTk.PhotoImage(image)
        canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)
        canvas.image = tk_image 

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Check if a camera index was provided as an argument
    camera_index = 0 
    if len(sys.argv) > 1:
        try:
            camera_index = int(sys.argv[1])
        except ValueError:
            print("Invalid camera index. Using default (0).")

    root = tk.Tk()
    try:
        app = DemoApp(root, camera_index)
        root.protocol("WM_DELETE_WINDOW", app.cleanup)
        root.mainloop()
    except ValueError as e:
        print(e)
