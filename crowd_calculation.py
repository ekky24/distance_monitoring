import cv2
import dlib
import imutils
import numpy as np
from distance_tracker import DistanceTracker
from box_tracker import BoxTracker

input_path = 'inputs/input_03.mp4'
output_path = 'outputs/output_03.avi'
model_path = 'model/MobileNetSSD_deploy.caffemodel'
proto_path = 'model/MobileNetSSD_deploy.prototxt'

writer = None
W = None
H = None
frame_counter = 0
skip_frames = 30
confidence_thresh = 0.4

classes = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

model = cv2.dnn.readNetFromCaffe(proto_path, model_path)
vs = cv2.VideoCapture(input_path)
bt = BoxTracker(maxDisappeared=20, maxDistance=50)
dt = DistanceTracker(thresh_distance=100, max_contact=30)

trackers = []
distanceTracker = {}

while True:
	frame = vs.read()[1]

	if(frame is None):
		break

	frame = imutils.resize(frame, width=500)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	# if the frame dimensions are empty, set them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	if writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(output_path, fourcc, 30, (W, H), True)

	rects = []

	if(frame_counter % skip_frames == 0):
		trackers = []

		blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
		model.setInput(blob)
		detections = model.forward()

		for i in range(detections.shape[2]):
			confidence = detections[0, 0, i, 2]

			if(confidence > confidence_thresh):
				idx = int(detections[0, 0, i, 1])

				if(classes[idx] != 'person'):
					continue

				box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
				(startX, startY, endX, endY) = box.astype("int")

				# START TRACKING
				tracker = dlib.correlation_tracker()
				rect = dlib.rectangle(startX, startY, endX, endY)
				tracker.start_track(rgb, rect)

				trackers.append(tracker)
				rects.append((startX, startY, endX, endY))
	else:
		for tracker in trackers:
			# UPDATE TRACKER
			tracker.update(rgb)
			pos = tracker.get_position()

			startX = int(pos.left())
			startY = int(pos.top())
			endX = int(pos.right())
			endY = int(pos.bottom())

			rects.append((startX, startY, endX, endY))

	# CALCULATE NEW CENTROID
	objects = bt.update(rects)
	distance_result, total_violation = dt.calculate(objects)

	# DRAW RECT AND ID
	for (objectID, box) in objects.items():
		startX, startY, endX, endY = box
		text = "ID {}".format(objectID)
		cv2.putText(frame, text, (box[0] + 10, box[1] + 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
	

		if str(objectID) in distance_result:
			cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 3)

	info = [
		("Total Violation", total_violation),
	]

	# loop over the info tuples and draw them on our frame
	for (i, (k, v)) in enumerate(info):
		text = "{}: {}".format(k, v)
		cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
			cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

	writer.write(frame)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

	frame_counter += 1

writer.release()
vs.release()
cv2.destroyAllWindows()