import cv2
import mediapipe as mp
import numpy as np
import time
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Define a function that calculates an angle between three points in x,y space
def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
    return angle 

# Specify the location of the file with the video to be read
filename_r = '/home/rafik/PROJECTS/pose1/Material/IMG_3620.MOV'
# Specify the location where the new video with detections will be written
filename_w = '/home/rafik/PROJECTS/pose1/Material/IMG_3620_test.mp4'
# Capture the video from the file
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
last_event_time = None
frame_count = 0

## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

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
            
            # Calculate angle
            angle = calculate_angle(hip, knee, ankle)
            
            # Visualize angle
            cv2.putText(image, f"{angle:.1f}", 
                           tuple(np.multiply(knee, frame_size).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            
            ##########################################################################################################################
            # Rep Counter logic
            # The rep is registered when the knee bend angle goes from the upper limit to the lower limit
            upper_limit = 160
            lower_limit = 80
            # Specify the minimum rep number to start registering the set
            min_rep_count = 2
            # Specify the minimum time between reps
            min_rep_time = 3 #[s]
            # Above the upper angle limit, register the "hold up" stage
            if angle >= upper_limit:
                stage = 'hold up'
            # Below the upper limit and after the "hold up" stage, register "down" stage
            if angle < upper_limit and stage == 'hold up':
                stage = 'down'
            # Below the lower limit and after the "down" stage, register the "hold down" stage
            if angle < lower_limit and stage =='down':
                stage = 'hold down'
            # Above the lower limit and after the "hold down" stage, register the "up" stage and count the rep
            if angle > lower_limit and stage == 'hold down':
                stage = 'up'
                rep_count +=1
                last_event_time = time
                print(last_event_time)

            # Register the real rep count only for sets with more than the min rep count
            if rep_count >= min_rep_count:
                real_counter = rep_count
            # Restart the counter when the reps do not repeat in less than the min rep time
            if time - last_event_time > min_rep_time:
                rep_count = 0
            ##########################################################################################################################

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
        
        # Display and write the video
        if ret == True:
            cv2.imshow('Mediapipe Feed', image)
            output.write(image)

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

