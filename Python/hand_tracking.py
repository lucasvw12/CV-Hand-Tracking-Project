import cv2
import mediapipe as mp
import math
import numpy as np
import time
import socket
capture = cv2.VideoCapture(0)


UDP_IP = "127.0.0.1"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


prev_point = None
pointing_frames = 0
fist_start_time = 0
GRAB_DISTANCE = 0.3


mp_hands = mp.solutions.hands # this is the hand tracking module that detects 21 landmarks
mp_drawing = mp.solutions.drawing_utils
hand = mp_hands.Hands()


def is_pinky_up(landmarks):
    pinky_tip = landmarks[20]
    pinky_dip = landmarks[19]
    pinky_pip = landmarks[18]
    pinky_mcp = landmarks[17]
    wrist = landmarks[0]


    is_up = (pinky_tip.y < pinky_dip.y < pinky_pip.y < pinky_mcp.y < wrist.y)
    index_curled = landmarks[8].y > landmarks[6].y
    middle_curled = landmarks[12].y > landmarks[10].y
    ring_curled = landmarks[16].y > landmarks[14].y
    return is_up and index_curled and middle_curled and ring_curled


def distance_to_circle(landmarks, circle):
    index_tip = landmarks[8]
    thumb_tip = landmarks[4]
    pinch_x = (index_tip.x + thumb_tip.x) / 2 * 640
    pinch_y = (index_tip.y + thumb_tip.y) / 2 * 480
    return math.sqrt((pinch_x - circle["x"])**2 + (pinch_y - circle["y"])**2)


def is_thumb_up(landmarks):
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    thumb_mcp = landmarks[2]
    thumb_cmc = landmarks[1]
    wrist = landmarks[0]


    is_up = (thumb_tip.y < thumb_ip.y < thumb_mcp.y < thumb_cmc.y < wrist.y)
    index_curled = landmarks[8].y > landmarks[6].y
    middle_curled = landmarks[12].y > landmarks[10].y
    ring_curled = landmarks[16].y > landmarks[14].y
    pinky_curled = landmarks[20].y > landmarks[18].y
    return is_up and index_curled and middle_curled and ring_curled and pinky_curled




def is_pointing(landmarks):
    index_extended = landmarks[8].y < landmarks[6].y
    middle_curled  = landmarks[12].y > landmarks[10].y
    ring_curled    = landmarks[16].y > landmarks[14].y
    pinky_curled   = landmarks[20].y > landmarks[18].y
    return index_extended and middle_curled and ring_curled and pinky_curled


fist_count = 0
def is_fist(landmarks):
    global fist_count
    count = 0
    for i in (5, 9, 13, 17):
        if (landmarks[i].y - landmarks[i+3].y) < 0.04:
            count += 1
    if count == 4:
        fist_count += 1
        if fist_count > 30: # Require fist to be held for ~1 second at 30 FPS
            return True    
    else:
        fist_count = 0
        return False


def dist(a, b):
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)


def find_circle(landmarks,circles):
    valid_circles = []
    for circle in circles:
        if distance_to_circle(landmarks, circle) < circle["radius"]:
            valid_circles.append(circle)
    return valid_circles


def closest_circle(landmarks, circles):
    closest = None
    valid_circles = find_circle(landmarks, circles)
    min_dist = float('inf')
    for circle in valid_circles:
        d = distance_to_circle(landmarks, circle)
        if d < min_dist:
            min_dist = d
            closest = circle
    return closest


def matrix_to_quaternion(matrix):
    m = matrix
    trace = m[0,0] + m[1,1] + m[2,2]
    if trace > 0:
        w = math.sqrt(trace + 1.0) * 0.5
        x = (m[1,2] - m[2,1]) / (4.0 * w)
        y = (m[2,0] - m[0,2]) / (4.0 * w)
        z = (m[0,1] - m[1,0]) / (4.0 * w)
    elif (m[0,0] > m[1,1]) and (m[0,0] > m[2,2]):
        x = math.sqrt(1.0 + m[0,0] - m[1,1] - m[2,2]) * 0.5
        w = (m[1,2] - m[2,1]) / (4.0 * x)
        y = (m[0,1] + m[1,0]) / (4.0 * x)
        z = (m[0,2] + m[2,0]) / (4.0 * x)
    elif m[1,1] > m[2,2]:
        y = math.sqrt(1.0 + m[1,1] - m[0,0] - m[2,2]) * 0.5
        w = (m[2,0] - m[0,2]) / (4.0 * y)
        x = (m[0,1] + m[1,0]) / (4.0 * y)
        z = (m[1,2] + m[2,1]) / (4.0 * y)
    elif m[2,2] > m[0,0]:
        z = math.sqrt(1.0 + m[2,2] - m[0,0] - m[1,1]) * 0.5
        w = (m[0,1] - m[1,0]) / (4.0 * z)
        x = (m[0,2] + m[2,0]) / (4.0 * z)
        y = (m[1,2] + m[2,1]) / (4.0 * z)
    return w, x, y, z


def is_rock_and_roll(landmarks):
    index_up = landmarks[8].y < landmarks[6].y
    pinky_up = landmarks[20].y < landmarks[18].y
    middle_curled = landmarks[12].y > landmarks[10].y
    ring_curled = landmarks[16].y > landmarks[14].y
    return index_up and pinky_up and middle_curled and ring_curled


def is_peace_sign(landmarks):
    index_up = landmarks[8].y < landmarks[6].y
    middle_up = landmarks[12].y < landmarks[10].y
    ring_curled = landmarks[16].y > landmarks[14].y
    pinky_curled = landmarks[20].y > landmarks[18].y
    return index_up and middle_up and ring_curled and pinky_curled


def get_hand_quaternion(landmarks):
    wrist = np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z])
    middle_mcp = np.array([landmarks[9].x, landmarks[9].y, landmarks[9].z])
    index_base = np.array([landmarks[5].x, landmarks[5].y, landmarks[5].z])
    pinky_base = np.array([landmarks[17].x, landmarks[17].y, landmarks[17].z])
   
    forward = middle_mcp - wrist
    side = pinky_base - index_base
    normal = np.cross(forward, side)


    forward = forward / np.linalg.norm(forward)
    side = side / np.linalg.norm(side)
    normal = normal / np.linalg.norm(normal)


    R = np.column_stack((forward, side, normal))
    quat = matrix_to_quaternion(R)
    return quat


def is_thumbs_up(landmarks):
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    thumb_mcp = landmarks[2]
    thumb_cmc = landmarks[1]
    wrist = landmarks[0]


    is_up = (thumb_tip.y < thumb_ip.y < thumb_mcp.y < thumb_cmc.y < wrist.y)
    index_curled = landmarks[8].y > landmarks[6].y
    middle_curled = landmarks[12].y > landmarks[10].y
    ring_curled = landmarks[16].y > landmarks[14].y
    pinky_curled = landmarks[20].y > landmarks[18].y
    return is_up and index_curled and middle_curled and ring_curled and pinky_curled


if not capture.isOpened():
    print("Error: Webcam could not be opened")
    exit()
frame_height = 480
frame_width = 640
canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
circles = []
pinching = False
long_pinching = False
active_circle = None
moving_circle = None
pinch_start_time = 0
initial_pinch_distance = 0
initial_circle_radius = 0
scaling_mode = False
scale_origin = None
tracking_mode = False
starting_position = None
throw_mode = False
start_throw_time = 0.0
throw_mode = False


delta = np.array([0.0, 0.0, 0.0])
scale_factor = 1.0
velocity = 0.0
while True:
    success, frame = capture.read()
    if not success:
        break
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hand.process(rgb_frame)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )
           
            landmarks = hand_landmarks.landmark
           
            if is_rock_and_roll(landmarks):
                if not tracking_mode:
                    tracking_mode = True
                    starting_position = landmarks[0]
            else:
                tracking_mode = False
            if tracking_mode:
                current_position = landmarks[0]
                delta = np.array([-3 * (current_position.x - starting_position.x), -3 * (current_position.y - starting_position.y), -3 * (current_position.z - starting_position.z)])


            if is_peace_sign(landmarks):
                if not scaling_mode:
                    scaling_mode = True
                    scale_origin = landmarks[0]
                else:
                    current_position = landmarks[0]
                    scale_delta = (-3 * (current_position.x - scale_origin.x))
                    scale_factor = 1 + scale_delta
            else:
                scaling_mode = False
           
            if is_thumbs_up(landmarks):
                print("Thumbs up detected")
                if not throw_mode:
                    throw_mode = True
                    start_throw_position = landmarks[4]
                    starting_throw_time = time.time()
                else:
                    current_time = time.time()
                    if current_time - starting_throw_time > 0.5:
                        print("Throw released")
                        final_position = landmarks[4]
                        distance = final_position.x - start_throw_position.x
                        velocity = distance / (current_time - starting_throw_time)
                        throw_mode = False
                        print(f"Throw velocity: {velocity:.2f}")
                       
                   
            quat = get_hand_quaternion(landmarks)
            w, x, y, z = quat
            message = f"{velocity},{scale_factor},{delta[0]},{delta[1]},{delta[2]},{-x:.4f},{-y:.4f},{z:.4f},{w:.4f}"
            sock.sendto(message.encode(), (UDP_IP, UDP_PORT))
            velocity = 0.0


            wrist = landmarks[0]
            middle_mcp = landmarks[9]
            hand_size = dist(wrist, middle_mcp)
           
            frame_width = 640
            frame_height = 480
           
            index_tip = hand_landmarks.landmark[8]
            thumb_tip = hand_landmarks.landmark[4]


            thumb_index_distance = dist(index_tip, thumb_tip)


            index_x_pix= index_tip.x * frame_width
            index_y_pix = index_tip.y * frame_height


            thumb_x_pix = thumb_tip.x * frame_width
            thumb_y_pix = thumb_tip.y * frame_height
            mid_x = (index_x_pix + thumb_x_pix) / 2
            mid_y = (index_y_pix + thumb_y_pix) / 2


            index_base = hand_landmarks.landmark[5]
            index_base_x_pix = index_base.x * frame_width
            index_base_y_pix = index_base.y * frame_height


            index_base_z = index_base.z
            index_tip_z = index_tip.z
           
            thumb_index_distance_pix = ((index_x_pix - thumb_x_pix) ** 2 + (index_y_pix - thumb_y_pix) ** 2) ** 0.5
            normalized_distance = thumb_index_distance / hand_size


            PINCH_THRESHOLD = 0.3
            if normalized_distance < PINCH_THRESHOLD:  
                is_pinch = True
            else:
                is_pinch = False
           
            PINCH_START = 0.30
            PINCH_RELEASE = 0.40
            LONG_PINCH_RELEASE = 1.25
            current_time = time.time()
            if not pinching:
                if normalized_distance < PINCH_START:
                    grabbed = None
                    for c in reversed(circles):
                        if c["frozen"] and distance_to_circle(landmarks, c) < c["radius"] * 1.5:
                            grabbed = c
                            break
                    if grabbed:
                        grabbed["frozen"] = False
                        moving_circle = grabbed
                        active_circle = grabbed
                        pinching = True
                        long_pinching = False
                        pinch_start_time = current_time
                    else:
                        new_circle = {"x":mid_x, "y": mid_y, "radius": 40, "frozen": False, "quaternion": get_hand_quaternion(landmarks)}
                        circles.append(new_circle)
                        active_circle = new_circle
                        moving_circle = None
                        pinching = True
                        long_pinching = False
                        pinch_start_time = current_time
            else:
                if active_circle is not None:
                    if not long_pinching:
                        if current_time - pinch_start_time > 3:
                            long_pinching = True
                            initial_pinch_distance = thumb_index_distance
                            initial_circle_radius = active_circle["radius"]
                    if moving_circle:
                        active_circle["x"] = mid_x
                        active_circle["y"] = mid_y
                    elif long_pinching:
                        scale_factor = thumb_index_distance / initial_pinch_distance
                        active_circle["radius"] = max(20, int(initial_circle_radius * scale_factor))                            
                        active_circle["x"] = mid_x
                        active_circle["y"] = mid_y
                    else:
                        if current_time - fist_start_time > 2:
                            active_circle["radius"] = max(20, int(thumb_index_distance_pix / 2))
                            active_circle["x"] = mid_x
                            active_circle["y"] = mid_y
                    if is_pinky_up(landmarks):
                        if current_time - fist_start_time > 2:
                            fist_start_time = current_time
                            active_circle["frozen"] = True
                           
                            pinching = False
                            long_pinching = False
                            active_circle = None
                            moving_circle = None
                    else:
                        release_threshold = LONG_PINCH_RELEASE if long_pinching else PINCH_RELEASE
                        if normalized_distance > release_threshold:    
                            circles.remove(active_circle)                    
                            long_pinching = False
                            pinching = False
                            moving_circle = None
                            active_circle = None


            pointing = is_pointing(landmarks)


            if pointing:
                pointing_frames += 1
            else:
                pointing_frames = 0
                prev_point = None


            if pointing and pointing_frames > 3:
                current_point = (int(index_x_pix), int(index_y_pix))
                if prev_point is not None:
                    cv2.line(canvas, prev_point, current_point, (0, 255, 0), thickness=5)
                prev_point = current_point


            if is_pinky_up(landmarks):
                if current_time - fist_start_time > 2:
                    fist_start_time = current_time
                    if active_circle is not None:
                        active_circle["frozen"] = True


                    pinching = False
                    long_pinching = False
                    active_circle = None
                    moving_circle = None


            if is_fist(landmarks):
                circles.clear()
                active_circle = None
                moving_circle = None
                pinching = False
                long_pinching = False
                fist_count = 0
                canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    frame = cv2.addWeighted(frame, 1.0, canvas, 0.6, 0)        


    for c in circles:
        color = (0, 200, 255) if c["frozen"] else (255, 80, 0)
        cv2.circle(frame, (int(c["x"]), int(c["y"])), c["radius"], color, -1)


    cv2.imshow("Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
capture.release()
cv2.destroyAllWindows()
sock.close()

