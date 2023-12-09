import sys
sys.path.append('/Users/jaas/Desktop/pythonenv/')
import cv2
import mediapipe as mp
import numpy as np
import time
import math

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


# Define a function that calculates an angle between three points in x,y space
def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle
    return angle


# Specify the location of the file with the video to be read
filename_r = '/Users/jaas/Downloads/squat1.mp4'
# Specify the location where the new video with detections will be written
filename_w = '/Users/jaas/Downloads/test.mp4'
# Capture the video from the file
cap = cv2.VideoCapture(filename_r)

# Get video frame dimensions and fps
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frame_size = (frame_width, frame_height)
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
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    while cap.isOpened():
        ret, frame = cap.read()

        # Count frames and calculate time in video
        frame_count += 1
        time = frame_count / fps  # [s]

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
            hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            hip1 = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee1 = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle1 = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            shoulder1 = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]

            arm = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            arm1 = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            heel = [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y]
            heel1 = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y]

            footindex = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]
            footindex1 = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]

            ARshoulder = (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * frame.shape[1])-150, int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * frame.shape[0]))
            ARshoulder1 = (int(ARshoulder[0]), int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * frame.shape[0]))

################################################################################################################################################################################################
            heel_dist = abs(heel[0] - heel1[0])
            foot_dist = abs(footindex[0] - footindex1[0])
            adjuster = heel_dist/foot_dist
            text_position = (int(heel[0] * frame.shape[1]), int(heel[1] * frame.shape[0]) - 20)
            text_position1 = (int(heel[0] * frame.shape[1]), int(heel[1] * frame.shape[0]))


            if adjuster >= 0.55:
                cv2.putText(image, "OK", text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(image, "NOT OK", text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if 0.99 <= (heel[1] / heel1 [1]) <= 1.01:
                cv2.putText(image, "OK", text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(image, "FIXHEEL", text_position1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

 


################################################################################################################################################################################################           
            ##Calculation of hip length
            hipdistance = abs(hip[0] - hip1[0])
            ##Distance between shoulder and arm
            shdistanceR = abs(shoulder[0] - arm [0])
            ##Distance between shoulder and arm right
            shdistanceL = abs(shoulder1[0] - arm1 [0])
            ###Tolerance setting for deviation at 10%
            Tolerance = 0.10
            acept_range = shdistanceR * Tolerance
            if abs(shdistanceR-shdistanceL) <= acept_range:
                print('OK')
                cv2.putText(image, f"{acept_range:.5f}",
                            tuple(np.multiply(arm1, frame_size).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA
                            )
                
                ###cv2.arrowedLine(image, ARshoulder, ARshoulder1, (255, 0, 0), 2)


            else:
                print('Adjust arm distance')
            ####Calculate angle for inclination to check if one side of barbell is higher######
            delta_x = abs(arm[0])-abs(arm1[0])
            delta_y = abs(arm[1])-abs(arm1[1])
            angle_radians = math.atan2(delta_y, delta_x)
            angle_degrees = math.degrees(angle_radians)
            arm_text_position = (int(arm[0] * frame.shape[1]), int(arm[1] * frame.shape[0]) - 20)


            # Ensure the angle is positive (0 to 360 degrees)
            if angle_degrees < 0:
                angle_degrees += 360
            angle_degrees = angle_degrees - 180
            # Check if the angle is below 2 degrees
            if -1 <= angle_degrees <= 1:
                cv2.putText(image, "OK", arm_text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                print("Positioning OK")
            else:
                cv2.putText(image, "NOK", arm_text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                print("Positioning NOTOK")
            cv2.putText(image, f"{angle_degrees:.1f}",
                        tuple(np.multiply(arm, frame_size).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA
                        )
            if angle_degrees < 0:
                cv2.arrowedLine(image, ARshoulder1, ARshoulder, (255, 0, 0), 2)
            else:
                cv2.arrowedLine(image, ARshoulder, ARshoulder1, (0, 0, 255), 2)


            ####
            # Calculate angle
            angle = calculate_angle(hip, knee, ankle)

            # Visualize angle
            cv2.putText(image, f"{angle:.1f}",
                        tuple(np.multiply(knee, frame_size).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
                        )
##########################################################################################################################
            # Rep Counter logic
            # The rep is registered when the knee bend angle goes from the upper limit to the lower limit
            upper_limit = 160
            lower_limit = 80
            # Specify the minimum rep number to start registering the set
            min_rep_count = 2
            # Specify the minimum time between reps
            min_rep_time = 3  # [s]
            # Above the upper angle limit, register the "hold up" stage
            if angle >= upper_limit:
                stage = 'hold up'
            # Below the upper limit and after the "hold up" stage, register "down" stage
            if angle < upper_limit and stage == 'hold up':
                stage = 'down'
            # Below the lower limit and after the "down" stage, register the "hold down" stage
            if angle < lower_limit and stage == 'down':
                stage = 'hold down'
            # Above the lower limit and after the "hold down" stage, register the "up" stage and count the rep
            if angle > lower_limit and stage == 'hold down':
                stage = 'up'
                rep_count += 1
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
        cv2.rectangle(image, (x_offset, y_offset), (390 + x_offset, 73 + y_offset), (128, 128, 128), -1)
        # Rep data
        cv2.putText(image, 'REPS', (15 + x_offset, 12 + y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(real_counter),
                    (10 + x_offset, 60 + y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        # Stage data
        cv2.putText(image, 'STAGE', (65 + x_offset, 12 + y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, stage,
                    (60 + x_offset, 60 + y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
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
