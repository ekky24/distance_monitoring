from scipy.spatial import distance as dist
import numpy as np

class DistanceTracker:
	def __init__(self, thresh_distance, max_contact):
		self.thresh_distance = thresh_distance
		self.max_contact = max_contact
		self.danger = {}
		self.violation = []

	def calculate(self, objects):
		object_keys = []
		result_danger = []

		for (objectID_1, box_1) in objects.items():
			for (objectID_2, box_2) in objects.items():
				key = str(objectID_1) + '-' + str(objectID_2)
				D = dist.euclidean(box_1, box_2)

				if(D == 0):
					continue

				if(D < self.thresh_distance):
					if(key not in self.danger):
						self.danger[key] = 0

					self.danger[key] += 1

					if(self.danger[key] > self.max_contact):
						ids = key.split('-')
						for idx in ids:
							if(idx not in result_danger):
								result_danger.append(idx)
							if(idx not in self.violation):
								self.violation.append(idx)
				else:
					if(key in self.danger):
						if(self.danger[key] > 0):
							self.danger[key] = 0

				# FOR REMOVING UNUSED KEYS
				object_keys.append(key)

		# REMOVE UNUSED OBJECT ID FROM SELF.DANGER
		keys = list(self.danger.keys())
		object_ids = list(objects.keys())

		for key in keys:
			if(key not in object_keys):
				del self.danger[key]

		# print(self.danger)
		return result_danger, len(self.violation)

