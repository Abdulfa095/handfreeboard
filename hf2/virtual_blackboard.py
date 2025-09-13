import cv2
import numpy as np
import mediapipe as mp
import time
from datetime import datetime

class VirtualBlackboard:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.drawing = False
        self.prev_x, self.prev_y = 0, 0
        self.canvas = None
        self.current_color = (0, 255, 0)  
        self.brush_thickness = 5
        self.colors = {
            'green': (0, 255, 0),
            'blue': (255, 0, 0),
            'red': (0, 0, 255),
            'white': (255, 255, 255),
            'yellow': (0, 255, 255)
        }
        self.color_names = list(self.colors.keys())
        self.current_color_index = 0
        self.gesture_cooldown = 0
        self.last_gesture_time = 0
        print("Virtual Blackboard Initialized!")
        print("Controls:")
        print("- Move index finger to draw")
        print("- Show palm (5 fingers up) to clear board")
        print("- Point with index finger up to change color")
        print("- Press 's' to save image")
        print("- Press 'q' to quit")
    def get_hand_landmarks(self, frame):
        """Extract hand landmarks from frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        return results
    def get_finger_position(self, landmarks, frame_shape):
        """Get index finger tip position"""
        h, w = frame_shape[:2]
        x = int(landmarks.landmark[8].x * w)
        y = int(landmarks.landmark[8].y * h)
        return x, y
    def count_raised_fingers(self, landmarks):
        """Count number of raised fingers for gesture detection"""
        fingers_up = 0
        if landmarks.landmark[4].x > landmarks.landmark[3].x:
            fingers_up += 1
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]
        for tip, pip in zip(finger_tips, finger_pips):
            if landmarks.landmark[tip].y < landmarks.landmark[pip].y:
                fingers_up += 1
        return fingers_up
    def detect_gestures(self, landmarks, current_time):
        """Detect hand gestures for controls"""
        if current_time - self.last_gesture_time < 2:
            return None
        fingers_up = self.count_raised_fingers(landmarks)
        if fingers_up == 5:
            self.last_gesture_time = current_time
            return "clear"
        elif fingers_up == 1 and landmarks.landmark[8].y < landmarks.landmark[6].y:
            self.last_gesture_time = current_time
            return "color_change"
        return None
    def draw_on_canvas(self, x, y):
        """Draw on the canvas"""
        if self.canvas is None:
            return
        if self.drawing and self.prev_x != 0 and self.prev_y != 0:
            cv2.line(self.canvas, (self.prev_x, self.prev_y), (x, y), 
                    self.current_color, self.brush_thickness)
        self.prev_x, self.prev_y = x, y
    def clear_canvas(self):
        """Clear the drawing canvas"""
        if self.canvas is not None:
            self.canvas.fill(0)
        print("Canvas cleared!")
    def change_color(self):
        """Cycle through available colors"""
        self.current_color_index = (self.current_color_index + 1) % len(self.colors)
        color_name = self.color_names[self.current_color_index]
        self.current_color = self.colors[color_name]
        print(f"Color changed to: {color_name}")
    def draw_ui(self, frame):
        """Draw user interface elements"""
        h, w = frame.shape[:2]
        color_name = self.color_names[self.current_color_index]
        cv2.rectangle(frame, (w-150, 10), (w-10, 50), self.current_color, -1)
        cv2.rectangle(frame, (w-150, 10), (w-10, 50), (255, 255, 255), 2)
        cv2.putText(frame, color_name.upper(), (w-140, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        instructions = [
            "Move finger to draw",
            "Palm up: Clear",
            "Point up: Change color",
            "Press 's': Save",
            "Press 'q': Quit"
        ]
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (10, 30 + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    def save_canvas(self):
        """Save the current canvas as an image"""
        if self.canvas is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"blackboard_{timestamp}.png"
            save_canvas = np.zeros_like(self.canvas)
            save_canvas[:] = (0, 0, 0)
            mask = np.any(self.canvas != [0, 0, 0], axis=-1)
            save_canvas[mask] = self.canvas[mask]
            cv2.imwrite(filename, save_canvas)
            print(f"Canvas saved as: {filename}")
        else:
            print("No canvas to save!")
    def run(self):
        """Main application loop"""
        print("\nStarting Virtual Blackboard...")
        print("Position your hand in front of the camera and start drawing!")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture video")
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            if self.canvas is None:
                self.canvas = np.zeros((h, w, 3), dtype=np.uint8)
            results = self.get_hand_landmarks(frame)
            current_time = time.time()
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    finger_x, finger_y = self.get_finger_position(hand_landmarks, frame.shape)
                    cv2.circle(frame, (finger_x, finger_y), 10, (255, 255, 0), -1)
                    gesture = self.detect_gestures(hand_landmarks, current_time)
                    if gesture == "clear":
                        self.clear_canvas()
                    elif gesture == "color_change":
                        self.change_color()
                    else:
                        fingers_up = self.count_raised_fingers(hand_landmarks)
                        if 1 <= fingers_up <= 2:
                            self.drawing = True
                            self.draw_on_canvas(finger_x, finger_y)
                        else:
                            self.drawing = False
                            self.prev_x, self.prev_y = 0, 0
            else:
                self.drawing = False
                self.prev_x, self.prev_y = 0, 0
            canvas_gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(canvas_gray, 1, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
            canvas_fg = cv2.bitwise_and(self.canvas, self.canvas, mask=mask)
            final_frame = cv2.add(frame_bg, canvas_fg)
            self.draw_ui(final_frame)
            cv2.imshow('Virtual Blackboard', final_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_canvas()
            elif key == ord('c'):
                self.clear_canvas()
        self.cap.release()
        cv2.destroyAllWindows()
        print("Virtual Blackboard closed!")
def main():
    """Main function to run the Virtual Blackboard"""
    try:
        blackboard = VirtualBlackboard()
        blackboard.run()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure your webcam is connected and working!")

if __name__ == "__main__":
    main()