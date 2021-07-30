import tempfile

import pandas as pd  # type: ignore
import streamlit as st  # type: ignore
import cv2  # type: ignore

from utils import download_link  # type: ignore

AOI = (
    "Other",
    "First AOI",
    "Second AOI",
    "Third AOI"
)

# Create persistent state variables
if 'annotation_dictionary' not in st.session_state:
    st.session_state.annotation_dictionary = {}
if 'last_choice' not in st.session_state:
    st.session_state.last_choice = "Other"
if 'video_capture' not in st.session_state:
    st.session_state.video_capture = None

st.title('Image annotation example')

if st.session_state.video_capture is None:
    # Read in a video
    uploaded_file = st.file_uploader("Upload video")
    if uploaded_file:
        with tempfile.NamedTemporaryFile() as tf:
            tf.write(uploaded_file.getvalue())
            st.session_state.video_capture = cv2.VideoCapture(tf.name)

if st.session_state.video_capture is not None:
    st.session_state.last_choice = st.sidebar.radio("Chose which AOI gaze is on", AOI)

    current_frame = int(st.session_state.video_capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1

    go_back_ten_frames = st.sidebar.button("Previous 10 frames")
    if go_back_ten_frames:
        st.session_state.video_capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame - 10)

    go_back_one_frame = st.sidebar.button("Previous frame")
    if go_back_one_frame:
        st.session_state.video_capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame - 1)

    go_to_next_frame = st.sidebar.button('Next Frame')
    if go_to_next_frame:
        tag = {current_frame: st.session_state.last_choice}
        st.session_state.annotation_dictionary.update(tag)

    go_to_next_ten_frames = st.sidebar.button("Next 10 frames")
    if go_to_next_ten_frames:
        st.session_state.video_capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame + 10)

    # Read the frame from video
    _, frame = st.session_state.video_capture.read()

    # Display the frame as image
    st.image(frame, caption=f'current frame id: {int(st.session_state.video_capture.get(cv2.CAP_PROP_POS_FRAMES))}',
             use_column_width=True)

    # Button to create a dataframe from the all_tags and display it
    show_dataframe = st.sidebar.button('Download data')
    if show_dataframe:
        # Create dataframe from dict and display
        df = pd.DataFrame(st.session_state.annotation_dictionary.items(), columns=["Frame_id", "Choice"])
        st.dataframe(df)
        # Download
        tmp_download_link = download_link(df, 'Annotated_data.csv', 'Click here to download your data!')
        st.markdown(tmp_download_link, unsafe_allow_html=True)

    show_dictionary = st.sidebar.button('Show raw data')
    if show_dictionary:
        st.write(st.session_state.get('annotation_dictionary'))
