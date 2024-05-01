from flask import Flask, request, render_template
import cv2
import numpy as np

app = Flask(__name__)

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

@app.route('/')
def upload_file():
    return render_template('upload.html')

def process_image(image):
    height, width, _ = image.shape
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (5, 5), 0)
    dilated = cv2.dilate(blur, np.ones((3, 3)))
    dilated_rgb = cv2.cvtColor(dilated, cv2.COLOR_GRAY2RGB)
    blob = cv2.dnn.blobFromImage(dilated_rgb, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    conf_threshold = 0.5
    nms_threshold = 0.4

    boxes = []
    confidences = []

    detected_cars = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold and classes[class_id] == "car":
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                overlap = False
                for car in detected_cars:
                    cx, cy, cw, ch = car
                    if x > cx and y > cy and x + w < cx + cw and y + h < cy + ch:
                        overlap = True
                        break

                if not overlap:
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    detected_cars.append((x, y, w, h))

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    car_count = 0
    if len(indices) > 0:
        for i in indices.flatten():
            box = boxes[i]
            x, y, w, h = box
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            car_count += 1

    return image, car_count

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            npimg = np.fromstring(file.read(), np.uint8)
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            result_img, car_count = process_image(img)
            cv2.imwrite('static/result.jpg', result_img)
            return render_template('result.html', car_count=car_count)
    return "Error"

if __name__ == '__main__':
    app.run(debug=True)
