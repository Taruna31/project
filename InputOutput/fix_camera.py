import cv2

# Load model and class labels
config_file = 'C:\\Users\\ASUS\\Desktop\\Pyside\\MainProject\\InputOutput\\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'C:\\Users\\ASUS\\Desktop\\Pyside\\MainProject\\InputOutput\\frozen_inference_graph.pb'
classlabels = []
file_name = 'C:\\Users\\ASUS\\Desktop\\Pyside\\MainProject\\InputOutput\\coco.names'
with open(file_name, 'rt') as fpt:
    classlabels = fpt.read().splitlines()

model = cv2.dnn_DetectionModel(frozen_model, config_file)

# Setting model parameters
model.setInputSize(320, 320)
model.setInputScale(1.8 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

# Camera setup
cap = cv2.VideoCapture(0)
cap.set(3, 600)
cap.set(4, 600)

while True:
    success, img = cap.read()

    # Detect objects
    classIds, confs, bbox = model.detect(img, confThreshold=0.5)

    # Draw detected objects
    if len(bbox) != 0:
        for classid, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color=(255, 0, 255), thickness=3)
            cv2.putText(img, classlabels[classid - 1].upper(), (box[0] + 10, box[1] + 20),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), thickness=2)

    # Display the image
    cv2.imshow("Object Detection", img)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()

