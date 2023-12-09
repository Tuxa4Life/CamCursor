import cv2
import mediapipe as mp
import math
import speech_recognition as sr
import pyttsx3
import pyautogui as pg


class VoiceAssistant():
    def __init__(self, micId=0):
        self.micId = micId
        self.listener = sr.Recognizer()
        self.engine = pyttsx3.init()

    def talk(self, text):
        self.engine.say(text)
        self.engine.runAndWait()

    def command(self):
        with sr.Microphone(device_index=self.micId) as source:
            self.talk('Listening')

            audio = self.listener.listen(source)
            try:
                command = self.listener.recognize_google(audio)
                command = command.lower()

                self.talk('Got it')
                if 'nothing' in command or 'cancel' in command or 'delete' in command:
                    pg.typewrite(' ')
                    return
                if 'copy' in command or 'remember' in command:
                    pg.hotkey('ctrl', 'c')
                    return
                if 'paste' in command:
                    pg.hotkey('ctrl', 'v')
                    return
                pg.typewrite(command)
            except:
                self.talk("Couldn't hear you well")


class HandDetector():
    def __init__(self, mode=False, maxHands=2, complexity=1, detectionConfidence=0.5, trackConfidence=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.complexity = complexity
        self.detectionConfidence = detectionConfidence
        self.trackConfidence = trackConfidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.complexity, self.detectionConfidence,
                                        self.trackConfidence)
        self.mpDraw = mp.solutions.drawing_utils

        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True, flip=False):
        if flip: img = cv2.flip(img, 1)

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLMs in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLMs, self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handId=0, draw=True, flip=False):
        if flip: img = cv2.flip(img, 1)

        self.lmList = []
        if self.results.multi_hand_landmarks:
            targetHand = self.results.multi_hand_landmarks[handId]
            for id, lm in enumerate(targetHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (230, 185, 4), cv2.FILLED)
        return self.lmList

    def fingersUp(self):
        fingers = []

        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def findDistance(self, p1, p2, img, draw=True, r=5, t=3, flip=False):
        if flip: img = cv2.flip(img, 1)

        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)

        return length, img


class FaceDetector():
    def __init__(self, cascade):
        # faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.cascade = cascade

    def findFace(self, img, draw=True, flip=False):
        if flip: img = cv2.flip(img, 1)

        faces = self.cascade.detectMultiScale(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1.1, 4)

        if draw:
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return img


class PoseDetection():
    def __init__(self):
        self.mpDraw = mpDraw = mp.solutions.drawing_utils
        self.mpPose = mpPose = mp.solutions.pose
        self.pose = mpPose.Pose()

    def findPose(self, img, draw=True, flip=False):
        if flip: img = cv2.flip(img, 1)

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(imgRGB)

        if draw:
            if results.pose_landmarks:
                self.mpDraw.draw_landmarks(img, results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img