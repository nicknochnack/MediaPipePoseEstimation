{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 0. Install and Import Dependencies"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "!pip install mediapipe opencv-python"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import cv2\n",
    "import mediapipe as mp\n",
    "import numpy as np\n",
    "mp_drawing = mp.solutions.drawing_utils\n",
    "mp_pose = mp.solutions.pose"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# VIDEO FEED\n",
    "cap = cv2.VideoCapture(0)\n",
    "while cap.isOpened():\n",
    "    ret, frame = cap.read()\n",
    "    cv2.imshow('Mediapipe Feed', frame)\n",
    "    \n",
    "    if cv2.waitKey(10) & 0xFF == ord('q'):\n",
    "        break\n",
    "        \n",
    "cap.release()\n",
    "cv2.destroyAllWindows()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 1. Make Detections"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "cap = cv2.VideoCapture(0)\n",
    "## Setup mediapipe instance\n",
    "with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:\n",
    "    while cap.isOpened():\n",
    "        ret, frame = cap.read()\n",
    "        \n",
    "        # Recolor image to RGB\n",
    "        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)\n",
    "        image.flags.writeable = False\n",
    "      \n",
    "        # Make detection\n",
    "        results = pose.process(image)\n",
    "    \n",
    "        # Recolor back to BGR\n",
    "        image.flags.writeable = True\n",
    "        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)\n",
    "        \n",
    "        # Render detections\n",
    "        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,\n",
    "                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), \n",
    "                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) \n",
    "                                 )               \n",
    "        \n",
    "        cv2.imshow('Mediapipe Feed', image)\n",
    "\n",
    "        if cv2.waitKey(10) & 0xFF == ord('q'):\n",
    "            break\n",
    "\n",
    "    cap.release()\n",
    "    cv2.destroyAllWindows()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "mp_drawing.DrawingSpec??"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 2. Determining Joints"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<img src=\"https://i.imgur.com/3j8BPdc.png\" style=\"height:300px\" >"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "cap = cv2.VideoCapture(0)\n",
    "## Setup mediapipe instance\n",
    "with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:\n",
    "    while cap.isOpened():\n",
    "        ret, frame = cap.read()\n",
    "        \n",
    "        # Recolor image to RGB\n",
    "        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)\n",
    "        image.flags.writeable = False\n",
    "      \n",
    "        # Make detection\n",
    "        results = pose.process(image)\n",
    "    \n",
    "        # Recolor back to BGR\n",
    "        image.flags.writeable = True\n",
    "        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)\n",
    "        \n",
    "        # Extract landmarks\n",
    "        try:\n",
    "            landmarks = results.pose_landmarks.landmark\n",
    "            print(landmarks)\n",
    "        except:\n",
    "            pass\n",
    "        \n",
    "        \n",
    "        # Render detections\n",
    "        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,\n",
    "                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), \n",
    "                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) \n",
    "                                 )               \n",
    "        \n",
    "        cv2.imshow('Mediapipe Feed', image)\n",
    "\n",
    "        if cv2.waitKey(10) & 0xFF == ord('q'):\n",
    "            break\n",
    "\n",
    "    cap.release()\n",
    "    cv2.destroyAllWindows()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "len(landmarks)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "for lndmrk in mp_pose.PoseLandmark:\n",
    "    print(lndmrk)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 3. Calculate Angles"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def calculate_angle(a,b,c):\n",
    "    a = np.array(a) # First\n",
    "    b = np.array(b) # Mid\n",
    "    c = np.array(c) # End\n",
    "    \n",
    "    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])\n",
    "    angle = np.abs(radians*180.0/np.pi)\n",
    "    \n",
    "    if angle >180.0:\n",
    "        angle = 360-angle\n",
    "        \n",
    "    return angle "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]\n",
    "elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]\n",
    "wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "shoulder, elbow, wrist"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "calculate_angle(shoulder, elbow, wrist)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "tuple(np.multiply(elbow, [640, 480]).astype(int))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "cap = cv2.VideoCapture(0)\n",
    "## Setup mediapipe instance\n",
    "with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:\n",
    "    while cap.isOpened():\n",
    "        ret, frame = cap.read()\n",
    "        \n",
    "        # Recolor image to RGB\n",
    "        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)\n",
    "        image.flags.writeable = False\n",
    "      \n",
    "        # Make detection\n",
    "        results = pose.process(image)\n",
    "    \n",
    "        # Recolor back to BGR\n",
    "        image.flags.writeable = True\n",
    "        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)\n",
    "        \n",
    "        # Extract landmarks\n",
    "        try:\n",
    "            landmarks = results.pose_landmarks.landmark\n",
    "            \n",
    "            # Get coordinates\n",
    "            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]\n",
    "            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]\n",
    "            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]\n",
    "            \n",
    "            # Calculate angle\n",
    "            angle = calculate_angle(shoulder, elbow, wrist)\n",
    "            \n",
    "            # Visualize angle\n",
    "            cv2.putText(image, str(angle), \n",
    "                           tuple(np.multiply(elbow, [640, 480]).astype(int)), \n",
    "                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA\n",
    "                                )\n",
    "                       \n",
    "        except:\n",
    "            pass\n",
    "        \n",
    "        \n",
    "        # Render detections\n",
    "        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,\n",
    "                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), \n",
    "                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) \n",
    "                                 )               \n",
    "        \n",
    "        cv2.imshow('Mediapipe Feed', image)\n",
    "\n",
    "        if cv2.waitKey(10) & 0xFF == ord('q'):\n",
    "            break\n",
    "\n",
    "    cap.release()\n",
    "    cv2.destroyAllWindows()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 4. Curl Counter"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "cap = cv2.VideoCapture(0)\n",
    "\n",
    "# Curl counter variables\n",
    "counter = 0 \n",
    "stage = None\n",
    "\n",
    "## Setup mediapipe instance\n",
    "with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:\n",
    "    while cap.isOpened():\n",
    "        ret, frame = cap.read()\n",
    "        \n",
    "        # Recolor image to RGB\n",
    "        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)\n",
    "        image.flags.writeable = False\n",
    "      \n",
    "        # Make detection\n",
    "        results = pose.process(image)\n",
    "    \n",
    "        # Recolor back to BGR\n",
    "        image.flags.writeable = True\n",
    "        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)\n",
    "        \n",
    "        # Extract landmarks\n",
    "        try:\n",
    "            landmarks = results.pose_landmarks.landmark\n",
    "            \n",
    "            # Get coordinates\n",
    "            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]\n",
    "            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]\n",
    "            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]\n",
    "            \n",
    "            # Calculate angle\n",
    "            angle = calculate_angle(shoulder, elbow, wrist)\n",
    "            \n",
    "            # Visualize angle\n",
    "            cv2.putText(image, str(angle), \n",
    "                           tuple(np.multiply(elbow, [640, 480]).astype(int)), \n",
    "                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA\n",
    "                                )\n",
    "            \n",
    "            # Curl counter logic\n",
    "            if angle > 160:\n",
    "                stage = \"down\"\n",
    "            if angle < 30 and stage =='down':\n",
    "                stage=\"up\"\n",
    "                counter +=1\n",
    "                print(counter)\n",
    "                       \n",
    "        except:\n",
    "            pass\n",
    "        \n",
    "        # Render curl counter\n",
    "        # Setup status box\n",
    "        cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)\n",
    "        \n",
    "        # Rep data\n",
    "        cv2.putText(image, 'REPS', (15,12), \n",
    "                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)\n",
    "        cv2.putText(image, str(counter), \n",
    "                    (10,60), \n",
    "                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)\n",
    "        \n",
    "        # Stage data\n",
    "        cv2.putText(image, 'STAGE', (65,12), \n",
    "                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)\n",
    "        cv2.putText(image, stage, \n",
    "                    (60,60), \n",
    "                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)\n",
    "        \n",
    "        \n",
    "        # Render detections\n",
    "        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,\n",
    "                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), \n",
    "                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) \n",
    "                                 )               \n",
    "        \n",
    "        cv2.imshow('Mediapipe Feed', image)\n",
    "\n",
    "        if cv2.waitKey(10) & 0xFF == ord('q'):\n",
    "            break\n",
    "\n",
    "    cap.release()\n",
    "    cv2.destroyAllWindows()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "mediapipe",
   "language": "python",
   "name": "mediapipe"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
