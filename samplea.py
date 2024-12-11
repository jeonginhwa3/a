import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
from io import BytesIO
import tempfile
import requests
import logging

# 로깅 설정
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# OpenCV 테스트
try:
    logger.info(f"OpenCV version: {cv2.__version__}")
except Exception as e:
    logger.error(f"OpenCV initialization failed: {e}")
    st.error(f"OpenCV initialization failed: {e}")

# ZeroDCE 모델 정의
class enhance_net_nopool(torch.nn.Module):
    # 생략 (원래 코드와 동일)

# 모델 로드 함수
@st.cache_resource
def load_models():
    # 생략 (원래 코드와 동일)

# 영상 프레임을 텐서로 변환
def preprocess_frame(frame):
    # 생략 (원래 코드와 동일)

# 비디오 처리
def process_video(input_video_path, enhancement_model, yolo_model):
    # 생략 (원래 코드와 동일)

# Streamlit UI
st.title("Object Detection & Brightness Enhancement")
st.write("Upload a video of a dark road to enhance brightness and detect objects.")

uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
if uploaded_video is not None:
    with st.spinner("Processing..."):
        enhancement_model, yolo_model = load_models()

        # Save uploaded video to a temporary file
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_input.write(uploaded_video.read())
        temp_input.close()

        output_path = process_video(temp_input.name, enhancement_model, yolo_model)

    st.video(output_path)
    with open(output_path, "rb") as file:
        st.download_button("Download Processed Video", file, file_name="output_video.mp4", mime="video/mp4")
