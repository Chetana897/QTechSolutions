import cv2
import numpy as np
from collections import OrderedDict

# --- Simple Centroid Tracker Implementation ---
class CentroidTracker:
    def __init__(self, maxDisappeared=40):
        # Unique ID for each object
        self.nextObjectID = 0
        # Dict to map object IDs to centroids
        self.objects = OrderedDict()
        # Tracks how many consecutive frames an object has disappeared
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        # When registering, add the centroid with a new object ID
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        # Remove an object ID from tracking
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        # If no rectangles (detections) are provided, mark existing objects as disappeared
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        # Compute centroids for each bounding box
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (x, y, w, h)) in enumerate(rects):
            cX = int(x + w / 2)
            cY = int(y + h / 2)
            inputCentroids[i] = (cX, cY)

        # If no objects are tracked, register all input centroids
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            # Grab the set of object IDs and their corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # Compute the distance matrix between tracked centroids and new centroids
            D = np.linalg.norm(np.array(objectCentroids)[:, np.newaxis] - inputCentroids, axis=2)

            # Find the smallest value in each row and then sort the row indexes
            rows = D.min(axis=1).argsort()
            # Find the corresponding column index for each row
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            # Loop over the (row, col) index pairs
            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)

            # Compute rows and columns that were not used
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # If the number of tracked objects is greater than or equal to new detections, mark disappeared
            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                # Register new objects for any unused columns
                for col in unusedCols:
                    self.register(inputCentroids[col])
        return self.objects

# --- YOLO Vehicle Detection Setup ---
# Load the YOLO network
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Get output layer names from the YOLO model
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Open the video file (or camera stream)
cap = cv2.VideoCapture("video.mp4")

# Initialize our centroid tracker and the set to store counted vehicle IDs
ct = CentroidTracker(maxDisappeared=40)
counted_ids = set()
vehicle_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape

    # Preprocess the frame and perform a forward pass through YOLO
    blob = cv2.dnn.blobFromImage(frame, scalefactor=0.00392, size=(416, 416),
                                 mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes = []
    confidences = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # Filter for cars (you can add more vehicle classes if needed)
            if confidence > 0.5 and classes[class_id] == "car":
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))

    # Apply Non-Maximum Suppression to remove overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)
    final_boxes = []
    if len(indexes) > 0:
        for i in indexes.flatten():
            final_boxes.append(boxes[i])
            # Draw the detection bounding box
            x, y, w, h = boxes[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Update the centroid tracker with the current frame detections
    objects = ct.update(final_boxes)

    # Loop over tracked objects and count new vehicles
    for objectID, centroid in objects.items():
        # Display the object ID near its centroid
        text = f"ID {objectID}"
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
        
        # Count the object if not already counted
        if objectID not in counted_ids:
            counted_ids.add(objectID)
            vehicle_count += 1

    # Display the total vehicle count on the frame
    cv2.putText(frame, f"Count: {vehicle_count}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
