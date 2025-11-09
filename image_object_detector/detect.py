from ultralytics  import YOLO

model = YOLO("yolov8n.pt")

image_path = "test_image/dog1.jpg"

results = model(image_path)

for result in results:
    for label, conf in zip(result.boxes.cls, result.boxes.conf):
        class_name = model.names[int(label)]
        print(f"Detected {class_name} with confidence {conf:.2f})")