import streamlit as st
import tensorflow as tf
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile  # Add this line
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import time



class_names = ['de-hydrated', 'hydrated']

pred_model = tf.keras.models.load_model("../WaterStressBestModels/inception.h5")

# Load a model
model = YOLO("../WaterStressBestModels/best-yolo.pt")  # load the custom model
# model = YOLO("yolov8n.pt")  # load an official model

def load_image(image_file):
    img = np.fromstring(image_file.read(), np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Add this line
    return img

def predict_image(img):
    results = model(img)  # predict on an image
    return results

def predict_frame(frame):
    results = model(frame)  # predict on a frame
    return results


def predict(roi, model=pred_model):


    image = cv2.resize(roi, (299, 299))


    image = image/255 # normalize the image in 0 to 1 range

    img_array = tf.expand_dims(image, 0)
    predictions = model.predict(img_array)

    print("Predictions:",predictions)

    predicted_class = class_names[int((predictions[0][0])+0.5)]
    if predicted_class == 'de-hydrated':
        confidence = 100 - round(100 * (predictions[0][0]), 2)
    else:
        confidence = round(100 * (predictions[0][0]), 2)


    return {predicted_class, confidence}



def plotImage(img, results):
    boxes = results[0].boxes.xyxy.numpy()  # format is (x1, y1, x2, y2, conf, cls)

    # Convert image from BGR to RGB
    #img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb = img


    # Draw bounding boxes and labels on the image
    for box in boxes:
        # Get box coordinates and class
        x1, y1, x2, y2 = box
        # label = labels[int(cls)]

        roi = img_rgb[int(y1):int(y2), int(x1):int(x2)]
        Conf, Class = predict(roi)
        if type(Class) is not str:
            x = Conf
            Conf = Class
            Class = x


        label = f"{Class[:11]}  {Conf}%"
        #label = f"{Conf}%"
        
        color = (0, 255, 0) if str(Class) == 'hydrated' else (0, 0, 255)
        # color = (255, 255, 0)
        print(Conf, Conf, Conf, Conf, Conf, Conf)


        # Draw rectangle (bounding box)
        cv2.rectangle(img_rgb, (int(x1), int(y1)), (int(x2), int(y2)), color, 5)

        # Put label near the rectangle
        cv2.putText(img_rgb, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Display the image
    return img_rgb




class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Predict with the model
        results = model(img)
        
        # Draw the prediction on the image
        # img = results[0].plot(show=False, save=False)
        img = plotImage(img, results)
        
        return img


if "upload" not in st.session_state:
    st.session_state["upload"] = 0

def change_state():
    st.session_state["upload"] = 1

def progress_bar():
    progress_bar = st.progress(0)
    for i in range(101):
        time.sleep(0.01)
        progress_bar.progress(i)
    time.sleep(1)

st.title("Water Stress Detection")
tab1, tab2, tab3 = st.tabs(["Image", "Video", "Live"])


def app():
    col1, col2 = st.columns([9,1])
    with tab1:
        image_file = st.file_uploader("Upload Image", type=['jpeg', 'png', 'jpg'], on_change=change_state)
        if image_file is not None:
            our_image = load_image(image_file)
            st.success("Photo uploaded succefully")
            st.text("Original Image")
            st.image(our_image)
            

            
            if st.button("Predict"):
                results = predict_image(our_image)
                if st.session_state["upload"] == 1:
                    progress_bar()
                    st.text("Prediction")
                    st.image(plotImage(our_image, results))
                    #st.image(results[0].plot(show=False, save=False))
                    #st.metric(label="Class", value="Hydrated", delta="90%")
    

    with tab2:
        video_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])
        if video_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())
            
            vf = cv2.VideoCapture(tfile.name)

            # Get the frames per second
            fps = vf.get(cv2.CAP_PROP_FPS)
            st.write(f"The FPS of the input video is {fps}")

            # Define the codec and create a VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

            stframe = st.empty()

            frame_count = 0  # Add this line

            while vf.isOpened():
                frame_count += 1  # Add this line
                ret, frame = vf.read()
                # if frame is read correctly ret is True
                if not ret:
                    break

                # Only make a prediction once every fps frames
                if frame_count % 1 == 0:  # Add this line
                    results = predict_frame(frame)
                    if results is not None:
                        frame = plotImage(frame, results)
                        # write the flipped frame
                        out.write(frame)
                    
                        stframe.image(frame, channels='BGR')
                        #stframe.image(frame)



            # Release everything if job is finished
            vf.release()
            out.release()

    with tab3:
        st.text("This is a real-time Monitoring Detection")
        webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)


if __name__ == '__main__':
    app()
