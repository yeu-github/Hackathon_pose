import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture("test.mp4")
# cap = cv2.VideoCapture("test.mp4")
# cap = cv2.VideoCapture("test.mp4")

# Sağ ve sol dirseklerin hareketini takip etmek için ayrı listeler oluşturun ve maksimum boyutu 10 olarak ayarlayın
right_elbow_track = []
left_elbow_track = []
max_track_size = 120

# #Video kaydedicisi oluşturun
video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter("output_video2.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (video_width, video_height))

x=0
a=0

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        _, image = cap.read()

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if x > 80:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            a = a + 1

        if results.pose_landmarks and a > 80:
            # Sağ dirsek
            right_joint_id = 13
            right_joint_position = results.pose_landmarks.landmark[right_joint_id]
            right_joint_x, right_joint_y = int(right_joint_position.x * image.shape[1]), int(right_joint_position.y * image.shape[0])
            right_elbow_track.append((right_joint_x, right_joint_y))

            # Sol el
            left_joint_id = 16
            left_joint_position = results.pose_landmarks.landmark[left_joint_id]
            left_joint_x, left_joint_y = int(left_joint_position.x * image.shape[1]), int(left_joint_position.y * image.shape[0])
            left_elbow_track.append((left_joint_x, left_joint_y))

            ## Listenin boyutunu maksimum boyutla sınırlayın
            if len(right_elbow_track) > max_track_size:
                right_elbow_track.pop(0)
            if len(left_elbow_track) > max_track_size:
                left_elbow_track.pop(0)

            # Son on frame'deki sağ dirsek hareketini takip eden çizgiyi çiz
            if len(right_elbow_track) > 1:
                for i in range(1, len(right_elbow_track)):
                    cv2.line(image, right_elbow_track[i - 1], right_elbow_track[i], (0, 255, 0), thickness=3)

            # Son on frame'deki sol dirsek hareketini takip eden çizgiyi çiz
            if len(left_elbow_track) > 1:
                for i in range(1, len(left_elbow_track)):
                    cv2.line(image, left_elbow_track[i - 1], left_elbow_track[i], (0, 0, 255), thickness=3)

        x = x + 1
        # Flip the image horizontally for a selfie-view display.
        # cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))

        # if cv2.waitKey(5) & 0xFF == 27:
        #     break
        out.write(cv2.flip(image, 1))
cap.release()
cv2.destroyAllWindows()
