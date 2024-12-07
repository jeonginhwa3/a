import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_closing, binary_opening, label, gaussian_filter, binary_erosion


# Streamlit 페이지 설정
st.set_page_config(layout="wide", page_title="Object Counting")
st.title("Automatic Object Counting")

# 작업 선택 옵션
task = st.sidebar.selectbox(
    "Select a task",
    ("Task 1: Count cookies on a bright background", 
     "Task 2: Count pens on a desk", 
     "Task 3: Count grains on a dark background")
)

# 파일 업로드
uploaded_file = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "tif"])


# Task 1: 밝은 배경에서 과자 개수 세기
def binarize_image_task1(image_array):
    """밝은 배경에서 물체를 분리하기 위한 이진화 처리 (과자 카운팅용)"""
    binary_image = (image_array < 180).astype(np.int8)  # 밝은 배경에서 어두운 객체 추출
    return binary_image


# Task 2: 책상 위에서 펜 개수 세기
def binarize_image_task2(image_array):
    """책상 위 펜 카운팅을 위한 이진화 처리"""
    binary_image = (image_array < 130).astype(np.int8)  # 어두운 배경에서 밝은 객체 추출
    return binary_image


# Task 1 & Task 2에 적용될 전처리
def morphological_processing(binary_image):
    """객체 내부 빈틈 메우기 및 노이즈 제거"""
    closed_image = binary_closing(binary_image, structure=np.ones((5, 5)))  # 클로징으로 빈틈 메우기
    opened_image = binary_opening(closed_image, structure=np.ones((5, 5)))  # 오프닝으로 노이즈 제거
    return opened_image


# Task 1 & Task 2: 객체 수 카운팅
def count_objects(binary_image, size_threshold=50):
    """객체 수 카운팅"""
    labeled_image, num_features = label(binary_image)
    object_sizes = np.bincount(labeled_image.flatten())
    valid_objects = object_sizes > size_threshold
    valid_labeled_image = np.where(np.isin(labeled_image, np.nonzero(valid_objects)[0]), labeled_image, 0)
    valid_object_count = len(np.unique(valid_labeled_image)) - 1  # 배경 제외
    return valid_labeled_image, valid_object_count


# Task 3: 쌀알 개수 세기 (두 번째 코드 통합)
def masking(image_np):
    """Task 3: 바이너리 마스크 생성"""
    f = np.array(image_np[:, :, 0])
    binary_image = (f > 70).astype(np.int8)
    return binary_image


def illumination_correction(image_np):
    """Task 3: 조명 보정"""
    blurred = gaussian_filter(image_np.astype(float), sigma=30)
    corrected_image = image_np - blurred
    corrected_image = np.clip(corrected_image, 0, 255).astype(np.uint8)
    return corrected_image


def erosion(binary_image):
    """Task 3: Erosion 연산"""
    eroded_image = binary_erosion(binary_image, structure=np.ones((3, 3))).astype(np.int8)
    return eroded_image


def connected(binary_image):
    """Task 3: 연결된 성분 분석"""
    labeled_image, num_features = label(binary_image)
    return labeled_image, num_features


# Main Process & Visualization
def process_and_display(uploaded_image):
    image = Image.open(uploaded_image)

    if task == "Task 1: Count cookies on a bright background":
        image_array = np.array(image.convert("L"))  # 흑백 변환
        binary_image = binarize_image_task1(image_array)
        processed_image = morphological_processing(binary_image)
        labeled_image, object_count = count_objects(processed_image, size_threshold=500)
        
        # 시각화
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(image, caption="Original Image", use_column_width=True)
        with col2:
            st.image(processed_image * 255, caption="Processed Binary Image", use_column_width=True, clamp=True)
        with col3:
            st.image(labeled_image, caption="Connected Components", use_column_width=True, clamp=True)
        st.subheader(f"Total number of cookies detected: {object_count}")

    elif task == "Task 2: Count pens on a desk":
        image_array = np.array(image.convert("L"))  # 흑백 변환
        binary_image = binarize_image_task2(image_array)
        processed_image = morphological_processing(binary_image)
        labeled_image, object_count = count_objects(processed_image, size_threshold=500)
        
        # 시각화
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(image, caption="Original Image", use_column_width=True)
        with col2:
            st.image(processed_image * 255, caption="Processed Binary Image", use_column_width=True, clamp=True)
        with col3:
            st.image(labeled_image, caption="Connected Components", use_column_width=True, clamp=True)
        st.subheader(f"Total number of pens detected: {object_count}")

    elif task == "Task 3: Count grains on a dark background":
        image_np = np.array(image)  # RGB 배열
        corrected_image = illumination_correction(image_np)
        binary_image = masking(corrected_image)
        eroded_image = erosion(binary_image)
        labeled_image, object_count = connected(eroded_image)

        # 시각화
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.image(image, caption="Original Image", use_column_width=True)
        with col2:
            st.image(corrected_image, caption="Illumination Corrected Image", use_column_width=True)
        with col3:
            st.image(eroded_image * 255, caption="Eroded Binary Mask", use_column_width=True, clamp=True)
        with col4:
            st.image(labeled_image, caption="Connected Components", use_column_width=True, clamp=True)
        st.subheader(f"Total number of grains detected: {object_count}")


if uploaded_file:
    process_and_display(uploaded_file)
