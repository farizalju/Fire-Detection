import streamlit as st
from ultralytics import YOLO
import cvzone
import cv2
import math
import numpy as np

# Define the Streamlit app
def main():
    st.title("Fire and Smoke Detection")

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi", "mkv"])

    if uploaded_file is not None:
        # Convert the file to an OpenCV image.
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)

        # Display the uploaded image.
        st.image(frame, channels="BGR")

        # Load the model
        model = YOLO('weights/best.pt')
        classnames = ['fire', 'smoke']

        # Run detection
        frame = cv2.resize(frame, (640, 480))
        result = model(frame, stream=True)

        for info in result:
            boxes = info.boxes
            for box in boxes:
                confidence = box.conf[0]
                confidence = math.ceil(confidence * 100)
                Class = int(box.cls[0])
                if confidence > 50:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    color = (0, 0, 255) if classnames[Class] == 'fire' else (255, 0, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 5)
                    cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 100],
                                       scale=1.5, thickness=2)

        # Convert the frame to RGB format for displaying in Streamlit
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the frame with detections in the Streamlit app
        st.image(frame, channels="RGB")

if __name__ == "__main__":
    main()
