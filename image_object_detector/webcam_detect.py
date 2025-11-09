from ultralytics import YOLO
import cv2
import time

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("❌ Cannot open camera. Try changing the index (0→1).")
    exit()

time.sleep(1)  # let camera warm up

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame not captured. Exiting...")
        break

    results = model(frame, verbose=False)

    for result in results:
        boxes = result.boxes.xyxy
        labels = result.boxes.cls
        conf = result.boxes.conf
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            label = model.names[int(labels[i])]
            confidence = conf[i]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    cv2.imshow("YOLO Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
