import cv2
import numpy as np
import TuxasModule as tm
import pyautogui as pg

wCam, hCam = 640, 480
wScr, hScr = pg.size()
frameR = 130
smoothing = 5
plocX, plocY = 0, 0
clocX, clocY = 0, 0

cap = cv2.VideoCapture(0)

detector = tm.HandDetector(maxHands=1)
voiceAss = tm.VoiceAssistant(micId=0)

mouseState = False


def mouseDown():
    global mouseState
    if mouseState:
        None
    else:
        pg.mouseDown()
        mouseState = True


def mouseUp():
    global mouseState
    if not mouseState:
        None
    else:
        pg.mouseUp()
        mouseState = False


while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    landmarkList = detector.findPosition(img, draw=False)

    if len(landmarkList) != 0:
        x1, y1 = landmarkList[8][1:]
        x2, y2 = landmarkList[4][1:]

        fingers = detector.fingersUp()

        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (200, 200, 0), 2)
        if fingers[1] and not fingers[2]:  # index - move mouse
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

            clocX = plocX + (x3 - plocX) / smoothing
            clocY = plocY + (y3 - plocY) / smoothing

            pg.moveTo(clocX, clocY)
            plocX = clocX
            plocY = clocY

        if fingers[1] and fingers[2] and not fingers[3] and not fingers[4]:  # index and middle - click
            length, img = detector.findDistance(8, 12, img)

            if length < 30:
                pg.click()

        if not fingers[0] and fingers[1] and not fingers[2] and not fingers[3] and fingers[4]:  # index and last - hold
            length, img = detector.findDistance(8, 20, img)

            if length > 90:
                mouseDown()
            else:
                mouseUp()

        if fingers[1] and fingers[2] and fingers[3] and not fingers[0] and not fingers[4]:  # index, middle and ring - right click
            pg.rightClick()

        if not fingers[1] and not fingers[2] and not fingers[3] and fingers[4]:  # thumb and last - voice assistant
            voiceAss.command()

        if fingers[2] and not fingers[1] and not fingers[3] and not fingers[4]:  # middle - quit
            quit()

    cv2.imshow("screw head", img)
    cv2.waitKey(1)