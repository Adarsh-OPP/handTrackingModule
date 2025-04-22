import cv2 as cv
import mediapipe as mp


class handDetector():
    def __init__(self, mode=False, maxHands=2, dectectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.dectectionCon = dectectionCon
        self.trackCon = trackCon
        
        self.mpHand = mp.solutions.hands
        self.hands = self.mpHand.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.dectectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, frame, draw=True):
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb)

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, hand_landmarks, self.mpHand.HAND_CONNECTIONS)

        return frame

    def findPosition(self, frame, handNo=0, draw=True):
        lmList = []
        
        if self.results.multi_hand_landmarks:
            myHand=self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id,cx,cy])
                if draw:
                    cv.circle(frame, (cx, cy), 6, (255, 0, 255), cv.FILLED)
        return lmList
    

def main():
    web_camp = cv.VideoCapture(0)
    detector = handDetector()
    while True:
        isTrue, frame = web_camp.read()
        frame = detector.findHands(frame)
        lmList = detector.findPosition(frame)
        if len(lmList) !=0:
            print(lmList[4])

        if not isTrue:
            break
        cv.imshow('web_camp', frame)
        if cv.waitKey(20) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()
