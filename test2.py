import cv2
from flask import Flask, Response, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import supervision as sv

app = Flask(__name__)
CORS(app)  # 啟用跨域支持

# 載入訓練好的 YOLO 模型
model = YOLO(f'model/best.pt')
bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# 停車格狀態初始化
parking_slots = [{"slot": i + 1, "status": "available"} for i in range(8)]

# 啟用攝影機
cap = cv2.VideoCapture(0)

# 設置解析度
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

# 檢查攝影機是否正常開啟
if not cap.isOpened():
    print("Unable to open camera")

def gen_frames():
    global parking_slots
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)

        detected_slots = []
        detected_cars = []

        for detection in detections:
            class_name = detection[5]["class_name"]

            if class_name.startswith("slot-"):
                box = detection[0]
                if len(box) == 4:
                    slot_number = int(class_name.split("-")[1])
                    detected_slots.append({"slot": slot_number, "box": box})

            elif class_name in ["car", "my-car"]:
                car_box = detection[0]
                if len(car_box) == 4:
                    detected_cars.append(car_box)

        update_parking_slots(detected_cars, detected_slots)

        annotated_image = bounding_box_annotator.annotate(scene=frame, detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

        ret, buffer = cv2.imencode(".jpg", annotated_image)
        if not ret:
            break

        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def update_parking_slots(detected_cars, detected_slots):
    global parking_slots
    for slot in parking_slots:
        slot["status"] = "available"

    for slot_info in detected_slots:
        slot_number = slot_info["slot"]
        slot_box = slot_info["box"]
        for car_box in detected_cars:
            if is_car_in_slot(car_box, slot_box, overlap_threshold=0.2):
                parking_slots[slot_number - 1]["status"] = "occupied"
                break

def is_car_in_slot(car_box, slot_position, overlap_threshold=0.4):
    x_min, y_min, x_max, y_max = car_box
    slot_x_min, slot_y_min, slot_x_max, slot_y_max = slot_position

    overlap_x = max(0, min(x_max, slot_x_max) - max(x_min, slot_x_min))
    overlap_y = max(0, min(y_max, slot_y_max) - max(y_min, slot_y_min))

    overlap_area = overlap_x * overlap_y
    slot_area = (slot_x_max - slot_x_min) * (slot_y_max - slot_y_min)

    overlap_ratio = overlap_area / slot_area
    print(f"Overlap ratio: {overlap_ratio:.2f}")
    return overlap_ratio > overlap_threshold

@app.route('/get_parking_slots', methods=['GET'])
def get_parking_slots():
    return jsonify(parking_slots)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)