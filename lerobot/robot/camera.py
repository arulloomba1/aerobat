import cv2
import time
import threading
from collections import deque

class Camera:
    def __init__(self, camera_id, resolution=(640, 480), fps=30):
        self.camera_id = camera_id
        self.resolution = resolution
        self.fps = fps
        self.cap = None
        self.buffer = deque(maxlen=10)  # Increased buffer size from 5 to 10
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self.last_frame_time = time.time()
        self.frame_count = 0
        self.fps_actual = 0
        self.start()

    def _process_frame(self, frame):
        try:
            if frame is None:
                return None
            # Convert to RGB if needed
            if len(frame.shape) == 2:  # Grayscale
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif frame.shape[2] == 4:  # RGBA
                frame = frame[:, :, :3]
            return frame
        except Exception as e:
            print(f"Error processing frame: {e}")
            return None

    def _capture_thread(self):
        while self.running:
            try:
                ret, frame = self.cap.read()
                if ret:
                    processed_frame = self._process_frame(frame)
                    if processed_frame is not None:
                        with self.lock:
                            self.buffer.append(processed_frame)
                            self.frame_count += 1
                            current_time = time.time()
                            if current_time - self.last_frame_time >= 1.0:
                                self.fps_actual = self.frame_count / (current_time - self.last_frame_time)
                                self.frame_count = 0
                                self.last_frame_time = current_time
                else:
                    print(f"Failed to read frame from camera {self.camera_id}")
                    time.sleep(0.1)  # Add small delay when frame read fails
            except Exception as e:
                print(f"Error in capture thread: {e}")
                time.sleep(0.1)  # Add small delay when error occurs 