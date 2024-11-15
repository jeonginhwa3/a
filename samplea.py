import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd
from scipy.ndimage import binary_erosion, binary_dilation, binary_opening

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

def binarize_image_task1(image_array):
    """밝은 배경에서 물체를 분리하기 위한 이진화 처리 (과자 카운팅용)"""
    binary_image = (image_array < 180).astype(np.int8)  # 임계값을 180으로 조정
    return binary_image

def binarize_image_task2(image_array):
    """책상 위 펜 카운팅을 위한 이진화 처리"""
    binary_image = (image_array < 130).astype(np.int8)
    return binary_image

def binarize_image_task3(image_array):
    """어두운 배경에서 물체를 분리하기 위한 이진화 처리 (곡물 카운팅용)"""
    binary_image = (image_array > 100).astype(np.int8)
    return binary_image

def opening(binary_image):
    """객체 분리와 노이즈 제거를 위한 opening 연산"""
    structure = np.ones((3, 3))  # 3x3 구조 요소 (핵심)
    opened_image = binary_opening(binary_image, structure=structure)
    return opened_image

def closing(binary_image):
    """객체 연결을 위한 closing 연산 (팽창 후 침식)"""
    structure = np.ones((3, 3))  # 3x3 구조 요소 (핵심)
    closed_image = binary_dilation(binary_image, structure=structure)
    closed_image = binary_erosion(closed_image, structure=structure)
    return closed_image

def post_process(binary_image):
    """이진화된 이미지에서 노이즈를 제거하는 후처리"""
    # opening을 통해 노이즈를 제거하고 객체 분리
    binary_image = opening(binary_image)  # Opening 연산 적용
    # closing을 통해 객체 연결 및 작은 구멍을 메운다
    binary_image = closing(binary_image)  # Closing 연산 적용
    # 객체의 크기를 최소화하여 작은 객체를 제거하는 방법
    binary_image = binary_erosion(binary_image)  # 침식(erosion)
    binary_image = binary_dilation(binary_image)  # 팽창(dilation)
    return binary_image

def count_objects(binary_image, size_threshold=50, task="Task 1"):
    """연결된 구성 요소 분석을 통한 객체 수 카운트 (작은 객체 제외)"""
    labeled_image, num_features = nd.label(binary_image)
    
    # Task 3인 경우에만 size_threshold를 50으로 설정
    if task == "Task 3: Count grains on a dark background":
        size_threshold = 30  # Task 3에 대해 크기 제한을 50으로 설정

    # 객체 크기 필터링 (예: size_threshold 픽셀 이상만 유효한 객체로 간주)
    object_sizes = np.bincount(labeled_image.flatten())
    valid_objects = object_sizes > 500  # size_threshold 크기 이상인 객체만 유효한 객체로 간주

    # 유효한 객체들만 남기기
    valid_labeled_image = np.where(np.isin(labeled_image, np.nonzero(valid_objects)[0]), labeled_image, 0)
    valid_object_count = len(np.unique(valid_labeled_image)) - 1  # 배경 제외

    return valid_labeled_image, valid_object_count

def process_and_display(uploaded_image):
    image = Image.open(uploaded_image)
    image_array = np.array(image.convert("L"))  # 흑백 변환 후 배열로 변환

    if task == "Task 1: Count cookies on a bright background":
        binary_image = binarize_image_task1(image_array)
    elif task == "Task 2: Count pens on a desk":
        binary_image = binarize_image_task2(image_array)
    elif task == "Task 3: Count grains on a dark background":
        binary_image = binarize_image_task3(image_array)

    # 후처리 적용
    binary_image = post_process(binary_image)

    # 연결된 구성 요소 분석 및 결과 시각화
    labeled_image, object_count = count_objects(binary_image, size_threshold=50, task=task)
    
    # 원본 이미지
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(image, caption="Original Image", use_column_width=True)

    # 이진화 이미지
    with col2:
        st.image(binary_image, caption="Binary Image", use_column_width=True, clamp=True)

    # 연결된 구성 요소 이미지
    with col3:
        st.image(labeled_image, caption="Connected Components", use_column_width=True, clamp=True)

    # 객체 수 출력
    st.subheader(f"Total number of objects detected (after removing small ones): {object_count}")

if uploaded_file:
    process_and_display(uploaded_file)