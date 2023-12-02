import cv2
import mediapipe as mp
import numpy as np
import sys
sys.path.insert(0,'../functions')
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
import os

# Specify the location of the file with the video to be read
# Capture frames from a saved video
filename_r = '../../Data/videos/exercise_stock_video3.mp4'
# filename_r = 4
# Specify the location where the new video witwwh detections will be written
# filename_w = '../Data/videos/exercise_stock_video3_wRepCount.mp4'

cap = cv2.VideoCapture(filename_r)

# Get video frame dimensions and fps
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frame_size = (frame_width,frame_height)
fps = cap.get(5)

# # Create a video writer object
# output = cv2.VideoWriter(filename_w, cv2.VideoWriter_fourcc(*'XVID'), fps, frame_size)

frame_count = 0

## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.9, min_tracking_confidence=0.5, enable_segmentation=True) as pose:

    fps = cap.get(5)
    frame_width  = int(cap.get(3))
    frame_height = int(cap.get(4))

    while cap.isOpened():
        ret, frame = cap.read()

        # Stop the code when the video is over
        if not ret:
            print('Failed to grab a frame')
            break

        # Count frames and calculate time in video
        frame_count += 1
        time = frame_count / fps #[s]

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Get coordinates
            hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            


        except:
            pass
        
        

        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )


        # Display and write the video
        if ret == True:
            cv2.imshow('Mediapipe Feed', image)
            # output.write(image)

            # Pause or stop the video when instructed
            key = cv2.waitKey(5)
            # Stop by pressing 'q'
            if key == ord('q'):
                break
            # Pause by pressing 'w', resume by pressing any other key
            if key == ord('w'):
                cv2.waitKey(-1)


    cap.release()
    cv2.destroyAllWindows()



