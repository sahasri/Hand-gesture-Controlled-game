from cvzone.HandTrackingModule import HandDetector
import cv2
import pyautogui
import time
import keyboardInput
import numpy as np
import matplotlib.pyplot as plt


# Output video dimensions (increased)
output_width = 1024
output_height = 768

# Calculate the dimensions of the squares
square_width = int(output_width / 3)
square_height = output_height

# Initialize video capture
cap = cv2.VideoCapture(0)

# Hand detector
detector = HandDetector(maxHands=1, detectionCon=0.8)

# Create background blur filter
background_blur_kernel_size = (21, 21)  # Adjust as needed
background_blur_sigma_x = 20  # Adjust as needed

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # to save the original video captured by the camera
    originalVideo = frame

    # Increase contrast and brightness for clarity enhancement
    alpha = 1  # Contrast control (1.0-3.0)
    beta = 2  # Brightness control (0-100)
    frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

    # resize the video
    frame = cv2.resize(frame, (output_width, output_height))

    height, width, channel = frame.shape

    # Create an overlay
    overlay = np.zeros_like(frame, dtype=np.uint8)

    # Draw squares on the overlay
    # Left square
    cv2.rectangle(overlay, (0, 0), (square_width, square_height), (255, 255, 255), -1)
    # Right square
    cv2.rectangle(overlay, (2 * square_width, 0), (output_width, square_height), (255, 255, 255), -1)

    # Combine the frame and overlay with opacity
    opacity = 0.2
    cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)

    # Flip the frame horizontally to fix mirroring
    frame = cv2.flip(frame, 1)

    # Convert BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Apply background blur
    background_blur = cv2.GaussianBlur(originalVideo, background_blur_kernel_size, background_blur_sigma_x)

    # hands
    hands, frame = detector.findHands(frame)

    data =[]
    if hands:

        # Convert hand coordinates in to one array
        hand = hands[0]
        lmList = hand['lmList']
        # print(lmList)
        for lm in lmList:
            data.extend([lm[0], 720 - lm[1], lm[2]])

        print(f'width : {width}')
        print(f'height : {height}')

        # Check if hand is open or closed
        # Hand open
        if data[12*3+1] > data[4*3+1]:
            cv2.putText(frame, "Forward", (400, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            keyboardInput.press_key('w')
            keyboardInput.release_key('s')
            if data[9 * 3] < square_width:  # Hand left
                cv2.putText(frame, "Left", (100, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
                keyboardInput.press_key('a')
            else:
                keyboardInput.release_key('a')

            if data[9 * 3] > square_width*2:  # Hand right
                cv2.putText(frame, "Right", (800, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
                keyboardInput.press_key('d')
            else:
                keyboardInput.release_key('d')

        # Hand closed
        else:
            cv2.putText(frame, "Backward", (405, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            keyboardInput.press_key('s')
            keyboardInput.release_key('w')
            if data[9 * 3] < square_width:  # Hand left
                cv2.putText(frame, "Left", (100, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
                keyboardInput.press_key('a')
            else:
                keyboardInput.release_key('a')

            if data[9 * 3] > square_width * 2:  # Hand right
                cv2.putText(frame, "Right", (800, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
                keyboardInput.press_key('d')
            else:
                keyboardInput.release_key('d')

    else:
        print('Show your hand to the camera')
        cv2.putText(frame, "No hand detected!", (300, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        # Release all keys
        keyboardInput.release_key('w')
        keyboardInput.release_key('s')
        keyboardInput.release_key('a')
        keyboardInput.release_key('d')

    # Display the frame
    cv2.imshow('Hand Detection', frame)

    # Display the original capturing video
    cv2.imshow('Original Video', originalVideo)

    # Blured Image
    # cv2.imshow('Hand blur', background_blur)

    # # Create subplots
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # fig.suptitle('Hand Detection')
    #
    # # Display the original capturing video
    # ax1.imshow(cv2.cvtColor(originalVideo, cv2.COLOR_BGR2RGB))
    # ax1.set_title('Original Video')
    #
    # # Display the frame with hand detection
    # ax2.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # ax2.set_title('Hand Detection')
    #
    # plt.show()

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
