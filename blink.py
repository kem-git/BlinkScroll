import cv2
import pyautogui
import time

# Load OpenCV's pre-trained eye detector
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

cap = cv2.VideoCapture(0)

# Parameters
BLINK_FRAMES = 3       # need N consecutive frames without eyes to count as blink
EYE_RESET_FRAMES = 3   # need N consecutive frames with eyes to reset blink state
COOLDOWN = 0.8         # seconds between scrolls
SCROLL_AMOUNT = -600

blink_active = False
closed_count = 0
open_count = 0
last_blink_time = 0

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)

    # Count frames with no eyes detected
    if len(eyes) == 0:
        closed_count += 1
        open_count = 0
    else:
        open_count += 1
        closed_count = 0

    # Blink detected
    if not blink_active and closed_count >= BLINK_FRAMES:
        now = time.time()
        if now - last_blink_time > COOLDOWN:
            print("Blink -> Scroll")
            pyautogui.scroll(SCROLL_AMOUNT)
            last_blink_time = now
            blink_active = True

    # Reset blink state only after eyes appear again consistently
    if blink_active and open_count >= EYE_RESET_FRAMES:
        blink_active = False

    # Draw detected eyes
    for (x, y, w, h) in eyes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Blink Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
