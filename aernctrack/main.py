from collections import deque
from imutils.video import VideoStream
from period.alg import get_period_alg
from period.fft import get_period_fft
from period.rnn import get_period_rnn
from util import clamp, shift_hsv
import config
import numpy as np
import argparse
import cv2
import imutils
import time
import matplotlib.pyplot as plt

# X: the set of times (X values)
X = []

# Y: the set of positions in one direction (either X or Y) (Y values)
Y = []

# If the object is in simple harmonic motion, the X and Y values should be sinusoidal,
# so an algorithm or model that can predict the period of a sinusoidal function can be
# used to predict the period of the object's motion.

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
args = vars(ap.parse_args())

color = (172, 56, 21)
thres = 50

lower_color = color
upper_color = color
start_time = time.time()


def reset(new_color):
    global color, lower_color, upper_color, thres, start_time
    # convert to int since it might be uint8
    new_color = (
        int(new_color[0]),
        int(new_color[1]),
        int(new_color[2])
    )
    color = new_color
    lower_color = shift_hsv(color, -thres)
    upper_color = shift_hsv(color, thres)
    start_time = time.time()
    print(' ** Reset time')
    X.clear()
    Y.clear()


reset(color)

pts = deque(maxlen=args["buffer"])

if not args.get("video", False):
    vs = VideoStream(src=1).start()
else:
    vs = cv2.VideoCapture(args["video"])
time.sleep(2.0)  # warmup


def click_event(event, x, y, flags, params):
    global color

    if event == cv2.EVENT_MOUSEWHEEL:
        global thres
        # update sensitivity on mouse wheel use
        up = flags > 0
        thres = clamp(thres + (1 if up else -1) * 5, 0, 255)
        print(f' *** Updated sensitivity to {thres}')
        reset(color)

    if event != cv2.EVENT_LBUTTONDOWN and event != cv2.EVENT_RBUTTONDOWN:
        return

    blurred = cv2.GaussianBlur(frame, (11, 11), 0)

    b = blurred[y, x, 0]
    g = blurred[y, x, 1]
    r = blurred[y, x, 2]

    bgr_color = np.uint8([[[b, g, r]]])
    hsv_color = cv2.cvtColor(bgr_color, cv2.COLOR_BGR2HSV)[0][0]

    print(f' ***  Updated color to RGB({r}, {g}, {b}) HSV({hsv_color[0]}, {hsv_color[1]}, {hsv_color[2]})')

    reset(hsv_color)


last_print = start_time

while True:
    frame = vs.read()
    # in case its a video
    frame = frame[1] if args.get("video", False) else frame
    if frame is None:
        # end of the video?
        break

    frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # mask for the color
    mask = cv2.inRange(hsv, lower_color, upper_color)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # find the contours
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None

    # only proceed if at least one contour was found
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        now = time.time()
        time_since_last_print = now - last_print
        if time_since_last_print > 0.1:
            t = now - start_time
            print(f'        ({float(int(t*10))/10}, {center[config.X_OR_Y]})')
            X.append(t)
            Y.append(center[config.X_OR_Y])
            last_print = now
        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
        cv2.circle(frame, center, 5, (0, 0, 255), -1)

    # update the points queue
    pts.appendleft(center)

    # loop over the set of tracked points
    for i in range(1, len(pts)):
        # if either of the tracked points are None, ignore
        # them
        if pts[i - 1] is None or pts[i] is None:
            continue
        # otherwise, compute the thickness of the line and
        # draw the connecting lines
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
    cv2.imshow(config.WINDOW_NAME, frame)
    cv2.setMouseCallback(config.WINDOW_NAME, click_event)

    # stop on 'q'
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break


if not args.get("video", False):
    vs.stop()
else:
    vs.release()
cv2.destroyAllWindows()

# plot the data
plt.plot(X, Y)
plt.show()

# print 'em results
print('\n*************************************')
print(f' * collected {len(X)} data points')
print(f' * period (alg): {get_period_alg(X, Y)}')
print(f' * period (fft): {get_period_fft(X, Y)}')
print(f' * period (rnn): {get_period_rnn(X, Y)}')
print('*************************************\n')
