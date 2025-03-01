import cv2
import ultralytics
from ultralytics import YOLO
import os
import numpy as np
import subprocess

print('OpenCV version:', cv2.__version__)

def loadModel():
    errors = []
    try:
        global model
        model = YOLO('../models/yolo11n.pt')
    except Exception as error:
        errors.append(str(error))
    if errors:
        print('---------------------------- Error printing start ----------------------------')
        for error in errors:
            print(error)
        print('---------------------------- Error printing ended ----------------------------')
    else:
        print("Model loaded successfully.")

def getClassColors(model):
    """ Generate distinct colors for each class based on the number of classes in the model """
    num_classes = len(model.names)
    np.random.seed(203154)  # Ensures consistent colors between runs
    colors = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
    return {i: (int(colors[i][0]), int(colors[i][1]), int(colors[i][2])) for i in range(num_classes)}

def compressIfBig(filePath):
    """ Compress video file if it's too big """
    if os.path.getsize(filePath) > 62914560:  # 60MB
        print(f"Compressing {filePath}...")
        subprocess.run(["ffmpeg", "-i", filePath, "-vcodec", "h264", "-acodec", "aac", "-strict", "experimental",
                        filePath.replace(".mp4", "_Compressed.mp4"), "-y"])
        print(f"Compressed file saved at: {filePath.replace('.mp4', '_Compressed.mp4')}")

if __name__ == "__main__":
    loadModel()
    class_colors = getClassColors(model)

    videoPath = "../test/1.mp4"
    outputPath = "../test/1_out.mp4"
    outputCompressedPath = "../test/1_comp.mp4"

    if not os.path.exists(videoPath):
        print("Error: File does not exist at the specified path.")
        exit()

    vCap = cv2.VideoCapture(videoPath)

    # Get video properties
    frame_width = int(vCap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vCap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vCap.get(cv2.CAP_PROP_FPS)
    frame_count = int(vCap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 30  # Default fallback FPS

    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Alternative codec
    out = cv2.VideoWriter(outputPath, fourcc, fps, (frame_width, frame_height))

    cv2.namedWindow("Video Processed by YOLO11n", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Video Processed by YOLO11n", 640, 480)

    frame_idx = 0  # Track processed frames

    while vCap.isOpened():
        ret, frame = vCap.read()
        if not ret:
            print("End of video or read error at frame", frame_idx)
            break

        frame_idx += 1

        results = model.predict(frame, conf=0.01, workers=20)
        prediction = results[0].boxes

        oranges = 0

        if prediction is not None and len(prediction) > 0:
            boxes = prediction.xyxy
            confidences = prediction.conf
            classes = prediction.cls

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                conf = confidences[i]
                cls = int(classes[i])
                label = f'{model.names[cls]} {conf:.2f}'

                color = class_colors.get(cls, (0, 255, 0))  # Default to green if not found

                # if cls == 49:
                #     cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                #     text_y = max(y1 - 10, 20)
                #     cv2.putText(frame, label, (x1, text_y), cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1, cv2.LINE_AA)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                text_y = max(y1 - 10, 20)
                cv2.putText(frame, label, (x1, text_y), cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1, cv2.LINE_AA)

                if cls == 49:
                    oranges += 1

        cv2.imshow("Video Processed by YOLO11n", frame)
        out.write(frame)

        print(f"Processing frame {frame_idx}/{frame_count}...", end="\r")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Ensure everything is properly closed
    print(f"\nProcessed {frame_idx} frames out of {frame_count}.")
    print(f"Total orages found: {oranges}")
    vCap.release()
    out.release()
    cv2.destroyAllWindows()
    compressIfBig(outputPath)