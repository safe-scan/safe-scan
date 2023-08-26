import time
import cv2
import torch
import torchvision
from PIL import Image
from torchvision.transforms import transforms as transforms
import utils
import pygame

pygame.init()

alarm_sound = pygame.mixer.Sound("alarm.mp3")

transform = transforms.Compose([transforms.ToTensor()])
model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True, num_keypoints=17)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device).eval()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
	print('Error while trying to read video. Please check path again')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

frame_count = 0
total_fps = 0

pose_history = []
pose_window = 5

while cap.isOpened():
	ret, frame = cap.read()
	if ret:
		pil_image = Image.fromarray(frame).convert('RGB')
		orig_frame = frame
		image = transform(pil_image)
		image = image.unsqueeze(0).to(device)
		start_time = time.time()
		with torch.no_grad():
			outputs = model(image)
		end_time = time.time()
		output_image = utils.draw_keypoints(outputs, orig_frame)
		fps = 1 / (end_time - start_time)
		total_fps += fps
		frame_count += 1
		wait_time = max(1, int(fps / 4))

		detected_pose = utils.detect_poses(outputs)
		pose_history.append(detected_pose)
		if len(pose_history) > pose_window:
			pose_history.pop(0)

		if frame_count % pose_window == 0:
			most_common_pose = max(set(pose_history), key=pose_history.count)
			if most_common_pose != "Normal":
				alarm_sound.play()

		cv2.imshow('Pose detection frame', output_image)
		if cv2.waitKey(wait_time) & 0xFF == ord('q'):
			break
	else:
		break

cap.release()
cv2.destroyAllWindows()
avg_fps = total_fps / frame_count
print(f"Average FPS: {avg_fps:.3f}")
