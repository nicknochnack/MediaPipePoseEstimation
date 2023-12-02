import cv2
import mediapipe as mp
import numpy as np
import sys
sys.path.insert(0,'../functions')
from rep_counter1 import rep_counter
from joint_angle_calc1 import calculate_angle
from joint_angle_calc1 import calculate_angle_3d
from stage_classifier1 import stage_classifier
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
import os
from plotly_3dview import plot_landmarks

# Specify the location of the file with the video to be read
# Capture frames from a saved video
filename_r = '../Data/videos/exercise_stock_video3.mp4'
# Capture frames from a webcam feed
# filename_r = 4
# Specify the location where the new video with detections will be written
filename_w = '../Data/videos/exercise_stock_video3_wRepCount.mp4'

cap = cv2.VideoCapture(filename_r)

# Get video frame dimensions and fps
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frame_size = (frame_width,frame_height)
fps = cap.get(5)

# Create a video writer object
output = cv2.VideoWriter(filename_w, cv2.VideoWriter_fourcc(*'XVID'), fps, frame_size)

# Initialize the Rep Counter variables
real_counter = 0
stage = None
rep_count = 0
last_event_time = 0
frame_count = 0

# Initialize the stage classifier variables
stage_class = None
imcap = cv2.VideoCapture(filename_r)
im_count = 0
if im_count == 0:
    success, image_upper = imcap.read()
    im_count += 1
image_middown = image_upper
image_lower = image_upper
image_midup = image_upper
i = 0
j = 0
k = 0
l = 0
exercise_type = 'Squat'

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
        if results.segmentation_mask is None:
            print('Empty mask')
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            landmarks_3d = results.pose_world_landmarks.landmark
            
            # Get coordinates
            hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            
            hip_3d = [landmarks_3d[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks_3d[mp_pose.PoseLandmark.LEFT_HIP.value].y,landmarks_3d[mp_pose.PoseLandmark.LEFT_HIP.value].z]
            knee_3d = [landmarks_3d[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks_3d[mp_pose.PoseLandmark.LEFT_KNEE.value].y,landmarks_3d[mp_pose.PoseLandmark.LEFT_KNEE.value].z]
            ankle_3d = [landmarks_3d[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks_3d[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,landmarks_3d[mp_pose.PoseLandmark.LEFT_ANKLE.value].z]
        
            # Calculate angle
            angle = calculate_angle(hip, knee, ankle)
            angle_3d = calculate_angle_3d(hip_3d, knee_3d, ankle_3d)
            
            # Visualize angle
            cv2.putText(image, f"{angle_3d:.0f}", 
                           tuple(np.multiply(knee, frame_size).astype(int)+10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            

            ##########################################################################################################################
            # # Rep Counter logic
            # # The rep is registered when the knee bend angle goes from the upper limit to the lower limit
            upper_limit = 160
            lower_limit = 80
            # Specify the minimum rep number to start registering the set
            min_rep_count = 2
            # Specify the minimum time between reps
            min_rep_time = 3
            # Run the function
            stage,rep_count,last_event_time,real_counter = rep_counter(angle,upper_limit,lower_limit,min_rep_count,min_rep_time,time,
                                                                       stage,rep_count,last_event_time,real_counter)
            ##########################################################################################################################

            
            # Run stage classifier
            stage_class, image_upper, image_middown, image_lower, image_midup, i,j,k,l = stage_classifier(angle, exercise_type, image, stage_class, image_upper,image_middown,image_lower,image_midup, i,j,k,l)
            print(stage_class)
            
            # Run image segmentation
            if results.segmentation_mask is not None:
                segmented_image = image.copy()
                tightness = 0.01
                condition = condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > tightness
                bg_image = np.zeros(image.shape, dtype=np.uint8)
                segmented_image = np.where(condition, segmented_image, bg_image)

        except:
            pass
        
        
        # Render Rep Counter
        # Set the position of the Rep Counter window
        x_offset = 50
        y_offset = 100
        # Setup status box
        cv2.rectangle(image, (x_offset,y_offset), (390+x_offset,73+y_offset), (128,128,128), -1)
        # Rep data
        cv2.putText(image, 'REPS', (15+x_offset,12+y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(real_counter), 
                    (10+x_offset,60+y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        # Stage data
        cv2.putText(image, 'STAGE', (65+x_offset,12+y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, stage, 
                    (60+x_offset,60+y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        

        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )


        # mp_drawing.plot_landmarks(results.pose_world_landmarks,  mp_pose.POSE_CONNECTIONS)     
        
        fig = plot_landmarks(mp_pose, results.pose_world_landmarks,  mp_pose.POSE_CONNECTIONS)

        # Display and write the video
        if ret == True:
            cv2.imshow('Mediapipe Feed', image)
            output.write(image)

            if image_lower is None:
                print('Stage classifier not working')
            else:
                cv2.imshow('Classifier',image_lower)
                # print([i,j,k,l])
                # print(angle_3d)

            if results.segmentation_mask is None:
                print('Segmentation not working')
            else:
                cv2.imshow('Segmentation',segmented_image)

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

# Output the interactive plotly figure to browser to view
# fig.show()


