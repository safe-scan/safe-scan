import cv2
import matplotlib

edges = [(0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), (5, 7), (7, 9), (5, 11), (11, 13), (13, 15), (6, 12),
	(12, 14), (14, 16), (5, 6)]


def detect_poses(outputs):
	for i in range(len(outputs[0]['keypoints'])):
		keypoints = outputs[0]['keypoints'][i].cpu().detach().numpy()
		if outputs[0]['scores'][i] > 0.9:
			keypoints = keypoints[:, :].reshape(-1, 3)
			if detect_boxing_pose(keypoints):
				return "Boxing Pose"
			elif detect_shooting_pose(keypoints):
				return "Shooting Pose"
	return "Normal"


def detect_boxing_pose(keypoints):
	left_shoulder = keypoints[5]
	right_shoulder = keypoints[2]
	left_hand = keypoints[7]
	right_hand = keypoints[4]
	left_elbow = keypoints[6]
	right_elbow = keypoints[3]
	threshold = 100

	is_left_arm_bent = left_hand[1] > left_elbow[1] > left_shoulder[1]
	is_right_arm_bent = right_hand[1] > right_elbow[1] > right_shoulder[1]

	are_hands_close = abs(left_hand[0] - right_hand[0]) < threshold and abs(left_hand[1] - right_hand[1]) < threshold

	if is_left_arm_bent and is_right_arm_bent and are_hands_close:
		return True
	else:
		return False


def detect_shooting_pose(keypoints):
	left_shoulder = keypoints[5]
	right_shoulder = keypoints[2]
	left_hand = keypoints[7]
	right_hand = keypoints[4]
	threshold = 100

	is_left_arm_straight = abs(left_hand[1] - left_shoulder[1]) < threshold  # Adjust the threshold as needed
	is_right_arm_straight = abs(right_hand[1] - right_shoulder[1]) < threshold  # Adjust the threshold as needed

	are_hands_close = abs(left_hand[0] - right_hand[0]) < threshold and abs(left_hand[1] - right_hand[1]) < threshold
	if is_left_arm_straight and is_right_arm_straight and are_hands_close:
		return True
	else:
		return False


def draw_keypoints(outputs, image):
	detected_pose = detect_poses(outputs)

	for i in range(len(outputs[0]['keypoints'])):
		keypoints = outputs[0]['keypoints'][i].cpu().detach().numpy()
		if outputs[0]['scores'][i] > 0.9:
			keypoints = keypoints[:, :].reshape(-1, 3)

			for p in range(keypoints.shape[0]):
				cv2.circle(image, (int(keypoints[p, 0]), int(keypoints[p, 1])), 3, (0, 0, 255), thickness=-1,
						   lineType=cv2.FILLED)

			for ie, e in enumerate(edges):
				rgb = matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0])
				rgb = rgb * 255
				cv2.line(image, (keypoints[e, 0][0].astype(int), keypoints[e, 1][0].astype(int)),
						 (keypoints[e, 0][1].astype(int), keypoints[e, 1][1].astype(int)), tuple(rgb), 2,
						 lineType=cv2.LINE_AA)

			cv2.putText(image, detected_pose, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

	return image
