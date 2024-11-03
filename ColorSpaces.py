import streamlit as st
import cv2
from PIL import Image
import numpy as np

# Load custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load the custom CSS file
local_css("custom_style.css")

# Function to convert color spaces
def convert_color_space(image, color_space):
    if color_space == 'Grayscale':
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif color_space == 'HSV':
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    elif color_space == 'BGR':
        return image  # No change needed
    elif color_space == 'Hot Map':
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.applyColorMap(gray_image, cv2.COLORMAP_HOT)
    elif color_space == 'Edge Map':
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Canny(gray_image, 100, 200)
    elif color_space == 'CMYK':
        return convert_to_cmyk(image)
    elif color_space == 'LAB':
        return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    else:
        return image

# CMYK conversion function
def convert_to_cmyk(image):
    # Convert image to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Normalize the RGB values to [0, 1]
    rgb_image = rgb_image.astype(float) / 255.0
    # Compute the K channel
    K = 1 - np.max(rgb_image, axis=2)
    # Compute the C, M, and Y channels
    C = (1 - rgb_image[..., 0] - K) / (1 - K + 1e-10)
    M = (1 - rgb_image[..., 1] - K) / (1 - K + 1e-10)
    Y = (1 - rgb_image[..., 2] - K) / (1 - K + 1e-10)
    # Stack the channels to form the CMYK image
    CMYK = (np.dstack((C, M, Y, K)) * 255).astype(np.uint8)
    return CMYK

# Main application
def main():
    st.title("Image Color Space Converter")
    st.sidebar.header("Settings")

    uploaded_file = (st.sidebar
                     .file_uploader("Choose an image...",
                                    type=["jpg", "jpeg", "png"]))

    color_space = (st.sidebar
                   .selectbox("Select Color Space",
                              ["RGB", "Grayscale", "HSV",
                               "Hot Map", "Edge Map", "CMYK", "LAB"]))

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                 caption='Original Image', use_column_width=True, width=300)

        st.sidebar.subheader("Adjustments")
        brightness = st.sidebar.slider("Brightness", 0, 100, 50)
        contrast = st.sidebar.slider("Contrast", 0, 100, 50)

        adjusted_image = cv2.convertScaleAbs(image, alpha=contrast/50, beta=brightness-50)

        converted_image = convert_color_space(adjusted_image, color_space)

        if color_space in ['Grayscale', 'Edge Map']:
            st.image(converted_image, caption=f'Image in {color_space}', use_column_width=True, width=300, channels="GRAY")
        elif color_space == 'CMYK':
            st.image(converted_image, caption=f'Image in {color_space}', use_column_width=True, width=300)
        else:
            st.image(cv2.cvtColor(converted_image, cv2.COLOR_BGR2RGB), caption=f'Image in {color_space}', use_column_width=True, width=300)

if __name__ == "__main__":
    main()
