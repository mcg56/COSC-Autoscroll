# /usr/bin/python3
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils.video import FPS
from imutils import face_utils
from threading import Thread
from pynput.mouse import Button, Controller # CUM
import numpy as np
import argparse
import time
import dlib
import cv2
import math


#-------------------------------------Globals---------------------------------------#


# Head tilt thresholds. These will be initialised at the start of the program.
PITCH_THRESHOLD = 0
ROLL_THRESHOLD = 0
HEAD_ROLL = []
HEAD_PITCH = []
mouse = Controller() # CUM

# Parameters for threshold initialisation
THRESHOLD_START_TIME = time.time()    # Measure when eyes are first detected, after period of 15 seconds, max EAR calc
THRESHOLDS_DETECTED = False # reset if no eyes detected over 30 seconds
THRESHOLD_MEASURE = True

# Parameters to reset the EAR and head tilt thresholds, if a face is not detected for some period of time.
# e.g. in the situation where the driver of the vehicle changes.
DURATION_RESET_START = time.time()
DURATION_RESET = False

ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0, help="index of webcam on system")
ap.add_argument("-a", "--alarm", type=str, default="beep-09.wav", help="path alarm .WAV file")
args = vars(ap.parse_args())


#----------------------------------Helper functions-----------------------------------#


def determine_headpose(frame, landmarks):
    """This function helps to calculate head pose, as a generalisation of the Perspective and Point problem (PnP).
    The cv2.solvePnP solves the equation as described in the report for the project, producing the rotation vector
    and translation vector for the standard head 3D points to the 2D image landmarks found.
    Correction for reprojection error occurs to produce the projection matrix, which is used to calculate angles
    for head roll, pitch and yaw.

    Note: assumes no lens distortion present (calibrated camera)

    Args:
        frame: The frame currently being processed. Used to approximate camera intrinsic parameters
        landmarks: The array of co-ordinates representing facial landmarks

    Returns:
        Four values
        imgpts: 2D image points
        modelpts: 3D model points
        (roll, pitch, yaw): tuple containing angles for head roll, pitch and yaw
    """
    size = frame.shape  # (height, width, color_channel)

    # The 2D projections of landmarks on the image
    image_points = np.array([
        landmarks[30],  # Nose tip
        landmarks[8],   # Chin
        landmarks[36],  # Left outer eye left corner
        landmarks[45],  # Right outer eye right corner
        landmarks[48],  # Left mouth corner
        landmarks[54]   # Right mouth corner
    ], dtype="double")

    # 3D standard model points, assuming a standard, forward facing model for the head.
    model_points = np.array([
        (0.0, 0.0, 0.0),          # Nose tip
        (0.0, -330.0, -65.0),     # Chin
        (-225.0, 170.0, -135.0),  # Left eye left corner
        (225.0, 170.0, -135.0),   # Right eye right corner
        (-150.0, -150.0, -125.0), # Left mouth corner
        (150.0, -150.0, -125.0)   # Right mouth corner
    ])

    # Camera internals
    center = (size[1] / 2, size[0] / 2)
    focal_length = center[0] / np.tan(60 / 2 * np.pi / 180)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    # Solve PnP equation for rotation and translation vectors
    dist_coeffs = np.zeros((4, 1))
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                  dist_coeffs)
    axis = np.float32([[500, 0, 0],
                       [0, 500, 0],
                       [0, 0, 500]])

    # Account for reprojection error, produce projection matrix
    imgpts, jac = cv2.projectPoints(axis, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    modelpts, jac2 = cv2.projectPoints(model_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    rvec_matrix = cv2.Rodrigues(rotation_vector)[0]

    # Determine angles for roll, pitch and yaw using the projection matrix.
    proj_matrix = np.hstack((rvec_matrix, translation_vector))
    eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]

    pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]

    roll = -math.degrees(math.asin(math.sin(roll)))
    pitch = math.degrees(math.asin(math.sin(pitch)))
    yaw = math.degrees(math.asin(math.sin(yaw)))    

    return imgpts, modelpts, (roll, pitch, yaw)


 
        
def head_tilt_threshold(rotate_degree):
    """Determines whether the observed head tilt exceeds thresholds for drowsiness"""
    global THRESHOLDS_DETECTED, HEAD_TILT_EXCEEDED, PITCH_LOW, PITCH_HI
    roll = rotate_degree[0]
    pitch = rotate_degree[1]
    yaw = rotate_degree[2]
    
    # Pitch hysteresis
    PITCH_LOW = 6
    PITCH_HI = 10
    

    if THRESHOLDS_DETECTED:
        if in_threshold(pitch, PITCH_THRESHOLD, 2):
            HEAD_TILT_EXCEEDED = False

        else:
            #print("head tilt above threshold")
            HEAD_TILT_EXCEEDED = True
            
            if (pitch >= PITCH_THRESHOLD + PITCH_LOW): #CUM
                mouse.scroll(0,.1)
                if (pitch >= PITCH_THRESHOLD + PITCH_HI):
                    mouse.scroll(0,.5)
                    
            elif (pitch <= PITCH_THRESHOLD - PITCH_LOW): #CUM
                mouse.scroll(0,-.1)     
                if (pitch <= PITCH_THRESHOLD + PITCH_HI):
                    mouse.scroll(0,-.5)                
                        


def in_threshold(measure, threshold, change):
    """Helper function to determine whether observed angle exceeds threshold"""
    result1 = (threshold - change) < measure
    result2 = measure < (threshold + change)
    return result1 and result2


def draw_headpose(frame, imgpts, headpose, nose_landmark):
    """Helper function to draw head pose information

    # ORDER ON SCREEN: ROLL PITCH YAW
    # ROLL - lateral head tilt (bringing ear closer to shoulders)
    # PITCH - frontal head tilt (nodding head)
    # YAW - sideways gaze (turning head)
    """
    cv2.line(frame, tuple(nose_landmark), tuple(imgpts[1].ravel()), (0, 0, 255), 3)  # RED - ROLL
    cv2.line(frame, tuple(nose_landmark), tuple(imgpts[0].ravel()), (0, 255, 0), 3)  # GREEN - PITCH
    cv2.line(frame, tuple(nose_landmark), tuple(imgpts[2].ravel()), (255, 0, 0), 3)  # BLUE - YAW

    for j in range(len(headpose)):
        if j == 0:
            colour = (255, 0, 0)
            tilt = "Roll"
        elif j == 1:
            colour = (0, 255, 0)
            tilt = "Pitch"
        else:
            colour = (0, 0, 255)
            tilt = "Yaw"
        cv2.putText(frame, (tilt + ' {:05.2f}').format(float(headpose[j])), (10, 30 + (50 * j)), cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    colour, thickness=2, lineType=2)


def draw_landmarks(frame):
    """Helper function to draw dots on detected facial landmarks"""
    for (x, y) in landmarks:
        cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

def draw_eye_contours(left_eye, right_eye):
    """Helper function to draw eye contours on the frame"""
    leftEyeHull = cv2.convexHull(left_eye)
    rightEyeHull = cv2.convexHull(right_eye)
    cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
    cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)


def initialise_thresholds(roll, pitch, yaw):
    """Initialise thresholds for head roll, pitch and yaw. Due to intersubject variation.

    Similarly, due to different camera angles and positionings, head roll, pitch and yaw must be initialised, as
    a perfect frontal facing view cannot be assumed. However, to prevent extremely abnormal thresholds occurring,
    it is assumed that the subject will be relatively forward facing, thereby discarding head roll, pitch and yaw
    measurements that are extremely wide.
    """
    global THRESHOLD_MEASURE, THRESHOLD_START_TIME, current_time, THRESHOLDS_DETECTED, YAW_THRESHOLD, ROLL_THRESHOLD, PITCH_THRESHOLD
    initialisation_duration = 20    # duration in seconds that the thresholds should be initialised over

    if not THRESHOLDS_DETECTED:
        cv2.putText(frame, "Initialising...", (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if not THRESHOLD_MEASURE:
            THRESHOLD_MEASURE = True
            THRESHOLD_START_TIME = time.time()
        else:
            current_time = time.time()
            if current_time - THRESHOLD_START_TIME >= initialisation_duration:
                THRESHOLDS_DETECTED = True
                THRESHOLD_MEASURE = False                
                

                if len(HEAD_ROLL) > 0 and len(HEAD_PITCH) > 0:
                    ROLL_THRESHOLD = sum(HEAD_ROLL) / len(HEAD_ROLL)
                    PITCH_THRESHOLD = sum(HEAD_PITCH) / len(HEAD_PITCH)                    
                    print("Subject pitch: " + str(PITCH_THRESHOLD))  # print head parameters
            else:
                # Obtain measurements every 2 seconds for head roll, pitch and yaw
                if (round(current_time - THRESHOLD_START_TIME, 1)) % 2 == 0:
                    # Discard values that are extremely wide, as we assume the subject is approximately forward facing.
                    if abs(roll) < 20:
                        HEAD_ROLL.append(roll)
                    if abs(pitch) < 20:
                        HEAD_PITCH.append(pitch)


def threshold_reset_handler():
    """Helper function to reset thresholds, if a face has not been detected for a sufficient amount of time"""
    global DURATION_RESET, DURATION_RESET_START, current_time, THRESHOLDS_DETECTED
    if not DURATION_RESET:
        DURATION_RESET = True
        DURATION_RESET_START = time.time()
    else:
        current_time = time.time()
        if current_time - DURATION_RESET_START >= 20:
            print("Face has not been detected for some time... Head tilt thresholds have been reset")
            THRESHOLDS_DETECTED = False
            DURATION_RESET = False


#-------------------------------------Main Program-------------------------------------#
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector() # The detector is used to localise facial regions in the frame
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")   # The predictor finds facial landmarks within the localised facial regions.

print("[INFO] starting video stream thread...")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)

fps = FPS().start()

## To save video file if required
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

while True:
    fps.update()
    frame = vs.read()
    frame = imutils.resize(frame, width=450)   # Resizing to a smaller frame size improves FPS, but does not allow for successful video recording on lab computers
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    face_regions = detector(gray, 0)    # localise facial regions

    # Assume that the largest face region represents the driver (the largest = closest to the camera), ignore other
    # detected faces
    max_face = 0
    max_width = 0
    face_found = False
    for face in face_regions:
        (x, y, w, h) = face_utils.rect_to_bb(face)
        #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)   # Uncomment this if you wish to draw a bounding box around detected faces

        if (w > max_width):
            max_width = w
            max_face = face
            face_found = True

    if face_found:
        DURATION_RESET = False
        blur = cv2.GaussianBlur(gray, (3,3), 0)
        # Gaussian blurring used to reduce noise, however the facial landmark detector performs adequately without Gaussian blurring.
        # Note: Thresholding and binarisation, plus morphological operations interfere with landmark detection.

        landmarks = predictor(blur, max_face)   # Detect facial landmarks
        landmarks = face_utils.shape_to_np(landmarks)   # convert to array for easier handling
        

        imgpts, _, headpose = determine_headpose(frame, landmarks)    # Determine headpose
        if THRESHOLDS_DETECTED == False:
            draw_headpose(frame, imgpts, headpose, landmarks[30])
            draw_landmarks(frame)

        

        initialise_thresholds(headpose[0], headpose[1], headpose[2])   # Initialise headpose thresholds

        # Drowsiness detection handlers
        head_tilt_threshold(headpose)

    # If we cannot detect a face, assume that the driver of the car may be changing, so reset thresholds after a
    # sufficient amount of time.
    else:
        threshold_reset_handler()

    cv2.imshow("Frame", frame)
    #out.write(frame)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        fps.stop()
        print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))         
        break
cv2.destroyAllWindows()
#out.release()
vs.stop()
vs.stream.release()
