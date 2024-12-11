import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
from io import BytesIO
import tempfile

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

# 모델 로드 함수
@st.cache_resource
def load_models():
    enhancement_model = enhance_net_nopool()
    enhancement_model.load_state_dict(torch.load("Iter_29000.pth", map_location=torch.device("cpu")))
    enhancement_model.eval()

    yolo_model = YOLO("yolov8n.pt")
    return enhancement_model, yolo_model

# 영상 프레임을 텐서로 변환
def preprocess_frame(frame):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    return transform(frame).unsqueeze(0)  # 배치 차원 추가

# 비디오 처리
def process_video(input_video_path, enhancement_model, yolo_model):
    cap = cv2.VideoCapture(input_video_path)
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    output_path = temp_output.name

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (256, 256))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO 객체 감지
        results = yolo_model(frame)
        detected_frame = results[0].plot() if results[0].boxes is not None else None

        # ZeroDCE 밝기 개선
        input_tensor = preprocess_frame(frame)
        with torch.no_grad():
            _, enhanced_frame = enhancement_model(input_tensor)

        enhanced_frame = enhanced_frame.squeeze(0).detach().numpy().transpose(1, 2, 0)
        enhanced_frame = np.clip(enhanced_frame, 0, 1) * 255
        enhanced_frame = enhanced_frame.astype(np.uint8)

        if detected_frame is not None:
            detected_frame_resized = cv2.resize(detected_frame, (256, 256))
            enhanced_frame_resized = cv2.resize(enhanced_frame, (256, 256))
            final_frame = cv2.addWeighted(detected_frame_resized, 0.7, enhanced_frame_resized, 0.3, 0)
            out.write(final_frame)
        else:
            enhanced_frame_resized = cv2.resize(enhanced_frame, (256, 256))
            out.write(enhanced_frame_resized)

    cap.release()
    out.release()
    return output_path

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

    st.video(output_path)  # 비디오 표시
    with open(output_path, "rb") as file:
        st.download_button("Download Processed Video", file, file_name="output_video.mp4", mime="video/mp4")
