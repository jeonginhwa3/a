import streamlit as st
import torch
import cv2
import numpy as np
from torchvision import transforms
from ultralytics import YOLO
import tempfile
from io import BytesIO
import requests

# ZeroDCE 모델 정의
class enhance_net_nopool(torch.nn.Module):
    def __init__(self):
        super(enhance_net_nopool, self).__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        number_f = 32
        self.e_conv1 = torch.nn.Conv2d(3, number_f, 3, 1, 1, bias=True)
        self.e_conv2 = torch.nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv3 = torch.nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv4 = torch.nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv5 = torch.nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv6 = torch.nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv7 = torch.nn.Conv2d(number_f * 2, 24, 3, 1, 1, bias=True)

    def forward(self, x):
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r, 3, dim=1)
        x = x + r1 * (torch.pow(x, 2) - x)
        x = x + r2 * (torch.pow(x, 2) - x)
        x = x + r3 * (torch.pow(x, 2) - x)
        enhance_image_1 = x + r4 * (torch.pow(x, 2) - x)
        x = enhance_image_1 + r5 * (torch.pow(enhance_image_1, 2) - enhance_image_1)
        x = x + r6 * (torch.pow(x, 2) - x)
        x = x + r7 * (torch.pow(x, 2) - x)
        enhance_image = x + r8 * (torch.pow(x, 2) - x)
        return enhance_image_1, enhance_image

# GitHub URL에서 모델 다운로드 및 로드
def load_enhancement_model(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to download file from {url}. HTTP Status: {response.status_code}")
    model = enhance_net_nopool()
    model.load_state_dict(torch.load(BytesIO(response.content), map_location=torch.device("cpu")))
    model.eval()
    return model

# YOLOv8 모델 로드
def load_yolo_model():
    model = YOLO("yolov8n.pt")  # YOLOv8 모델 사용
    return model

# 프레임 전처리
def preprocess_frame(frame):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    return transform(frame).unsqueeze(0)

# 비디오 처리 함수
def process_video(input_path, output_path, yolo_model, enhancement_model):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (256, 256))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo_model(frame)
        confidence_threshold = 0.5
        detected_frame = results[0].plot() if results[0].boxes is not None else None

        input_tensor = preprocess_frame(frame)
        _, enhanced_frame = enhancement_model(input_tensor)
        enhanced_frame = enhanced_frame.squeeze(0).detach().numpy().transpose(1, 2, 0)
        enhanced_frame = np.clip(enhanced_frame, 0, 1) * 255
        enhanced_frame = enhanced_frame.astype(np.uint8)

        if detected_frame is not None:
            detected_frame_resized = cv2.resize(detected_frame, (256, 256))
            enhanced_frame_resized = cv2.resize(enhanced_frame, (256, 256))
            if detected_frame_resized.shape[2] != enhanced_frame_resized.shape[2]:
                enhanced_frame_resized = cv2.cvtColor(enhanced_frame_resized, cv2.COLOR_GRAY2BGR)
            final_frame = cv2.addWeighted(detected_frame_resized, 0.7, enhanced_frame_resized, 0.3, 0)
            out.write(final_frame)
        else:
            enhanced_frame_resized = cv2.resize(enhanced_frame, (256, 256))
            out.write(enhanced_frame_resized)

    cap.release()
    out.release()

# Streamlit 앱
st.title("Object Detection & Brightness Enhancement")
st.write("Upload a video to process it with YOLO and ZeroDCE models.")

uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi"])
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as temp_input:
        temp_input.write(uploaded_file.read())
        input_video_path = temp_input.name

    output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

    # 모델 로드
    st.write("Loading models...")
    enhancement_model = load_enhancement_model("https://github.com/jeonginhwa3/a/raw/refs/heads/main/Iter_29000.pth")
    yolo_model = load_yolo_model()

    # 비디오 처리
    st.write("Processing video...")
    process_video(input_video_path, output_video_path, yolo_model, enhancement_model)

    # 비디오 출력 (재생)
    st.video(output_video_path)

    # 비디오 다운로드
    with open(output_video_path, "rb") as output_file:
        st.download_button("Download Processed Video", output_file, "processed_video.mp4", "video/mp4")

