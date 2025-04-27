import cv2
import time

def view_cameras():
    # Initialize cameras
    caps = []
    for i in range(2):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Camera {i} opened successfully")
            caps.append(cap)
        else:
            print(f"Could not open camera {i}")
    
    if not caps:
        print("No cameras could be opened")
        return
    
    print("Press 'q' to quit")
    
    try:
        while True:
            frames = []
            for i, cap in enumerate(caps):
                ret, frame = cap.read()
                if ret:
                    frames.append((i, frame))
                else:
                    print(f"Error reading from camera {i}")
            
            if not frames:
                print("No frames could be read from any camera")
                break
                
            # Display frames
            for i, frame in frames:
                cv2.imshow(f'Camera {i}', frame)
            
            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            # Small delay to prevent high CPU usage
            time.sleep(0.01)
    
    finally:
        # Release resources
        for cap in caps:
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    view_cameras() 