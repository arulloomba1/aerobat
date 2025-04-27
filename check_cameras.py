from lerobot.common.robot_devices.cameras.opencv import find_cameras

cameras = find_cameras()
print("\nAvailable cameras:")
for cam in cameras:
    print(f"Camera {cam['index']}: {cam.get('name', 'Unknown name')}") 