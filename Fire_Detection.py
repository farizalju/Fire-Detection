import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image
import tempfile

@st.cache_resource
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path="weights/last.pt", force_reload=True)
    return model

demo_img = "fire.9.png"
demo_video = "Fire_Video.mp4"

st.title('Fire Detection')
st.sidebar.title('App Mode')

app_mode = st.sidebar.selectbox('Choose the App Mode', ['About App', 'Run on Image', 'Run on Video', 'Run on WebCam'])

if app_mode == 'About App':
    st.subheader("About")
    st.markdown("<h5>This is the Fire Detection App created with custom trained models using YoloV5</h5>", unsafe_allow_html=True)
    st.markdown("- <h5>Select the App Mode in the SideBar</h5>", unsafe_allow_html=True)
    st.image("Images/first_1.png")
    st.markdown("- <h5>Upload the Image and Detect the Fires in Images</h5>", unsafe_allow_html=True)
    st.image("Images/second_2.png")
    st.markdown("- <h5>Upload the Video and Detect the fires in Videos</h5>", unsafe_allow_html=True)
    st.image("Images/third_3.png")
    st.markdown("- <h5>Live Detection</h5>", unsafe_allow_html=True)
    st.image("Images/fourth_4.png")
    st.markdown("- <h5>Click Start to start the camera</h5>", unsafe_allow_html=True)
    st.markdown("- <h5>Click Stop to stop the camera</h5>", unsafe_allow_html=True)
    st.markdown("""
                ## Features
- Detect on Image
- Detect on Videos
- Live Detection
## Tech Stack
- Python
- PyTorch
- Python CV
- Streamlit
- YoloV5
""")

if app_mode == 'Run on Image':
    st.subheader("Detected Fire:")
    text = st.markdown("")
    
    st.sidebar.markdown("---")
    img_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if img_file:
        image = np.array(Image.open(img_file))
    else:
        image = np.array(Image.open(demo_img))
        
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Original Image**")
    st.sidebar.image(image)
    
    model = load_model()
    results = model(image)
    length = len(results.xyxy[0])
    output = np.squeeze(results.render())
    text.write(f"<h1 style='text-align: center; color:red;'>{length}</h1>", unsafe_allow_html=True)
    st.subheader("Output Image")
    st.image(output, use_column_width=True)
    
if app_mode == 'Run on Video':
    st.subheader("Detected Fire:")
    text = st.markdown("")
    
    st.sidebar.markdown("---")
    
    st.subheader("Output")
    stframe = st.empty()
    
    video_file = st.sidebar.file_uploader("Upload a Video", type=['mp4', 'mov', 'avi', 'asf', 'm4v'])
    st.sidebar.markdown("---")
    tffile = tempfile.NamedTemporaryFile(delete=False)
    
    if not video_file:
        vid = cv2.VideoCapture(demo_video)
        tffile.name = demo_video
    else:
        tffile.write(video_file.read())
        vid = cv2.VideoCapture(tffile.name)
    
    st.sidebar.markdown("**Input Video**")
    st.sidebar.video(tffile.name)
    
    model = load_model()
    while vid.isOpened():
        ret, frame = vid.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame)
        length = len(results.xyxy[0])
        output = np.squeeze(results.render())
        text.write(f"<h1 style='text-align: center; color:red;'>{length}</h1>", unsafe_allow_html=True)
        stframe.image(output)

if app_mode == 'Run on WebCam':
    st.subheader("Detected Fire:")
    text = st.markdown("")
    
    st.sidebar.markdown("---")
    
    st.subheader("Output")
    stframe = st.empty()
    
    start = st.sidebar.button("Start")
    stop = st.sidebar.button("Stop")
    
    camera_source = st.sidebar.selectbox("Select Camera Source", ["Default Camera", "External Camera"])

    if start:
        cam_index = 0 if camera_source == "Default Camera" else 1
        cam = cv2.VideoCapture(cam_index)
        if not cam.isOpened():
            st.error(f"Failed to open camera {cam_index}.")
        else:
            model = load_model()
            while cam.isOpened():
                ret, frame = cam.read()
                if not ret:
                    st.error("Failed to capture image from webcam.")
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = model(frame)
                length = len(results.xyxy[0])
                output = np.squeeze(results.render())
                text.write(f"<h1 style='text-align: center; color:red;'>{length}</h1>", unsafe_allow_html=True)
                stframe.image(output)
                
                if stop:
                    cam.release()
                    break
