import streamlit as st
from PIL import Image, ImageDraw  # Import ImageDraw for drawing on the image
import numpy as np
import easyocr
import imageio

def pil_to_np(image):
    return np.array(image)

def draw_polylines(image, points, is_closed, color, thickness):
    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)
    draw.line(points, fill=color, width=thickness, joint="curve" if is_closed else None)
    return pil_to_np(img_pil)

def text_extraction_app():
    st.title("Image Text Extraction App")

    allowed_extensions = ["jpg", "jpeg", "png", "gif"]
    uploaded_file = st.file_uploader("Choose an image...", type=allowed_extensions)

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Convert the image to a NumPy array using imageio
        img_array = pil_to_np(image)

        # Perform text extraction using easyocr
        reader = easyocr.Reader(['en'])
        results = reader.readtext(img_array)

        # Extracted text
        text = " ".join(result[1] for result in results)

        # Display the extracted text
        st.subheader("Extracted Text:")
        st.text(text)

def text_detection_app():
    st.title("Text Detection and Bounding Boxes App")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "gif"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Convert the image to NumPy array using imageio
        img_array = pil_to_np(image)

        # Perform text detection and draw bounding boxes using easyocr with GPU acceleration
        reader = easyocr.Reader(['en'], gpu=True)  # Enable GPU acceleration
        results = reader.readtext(img_array)

        # Draw bounding boxes on the image
        for result in results:
            points = result[0]
            box = np.array(points).astype(np.int32).reshape((-1, 1, 2))
            img_array = draw_polylines(img_array, box, is_closed=True, color=(0, 255, 0), thickness=2)

        # Display the image with bounding boxes
        st.image(img_array, caption="Text Detection with Bounding Boxes", use_column_width=True)

def main():
    st.set_page_config(page_title="Combined App", page_icon=":rocket:")

    app_mode = st.sidebar.selectbox("Select App Mode", ["Text Extraction", "Text Detection"])

    if app_mode == "Text Extraction":
        text_extraction_app()
    elif app_mode == "Text Detection":
        text_detection_app()

if __name__ == "__main__":
    main()
