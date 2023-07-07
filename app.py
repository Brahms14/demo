from PIL import Image,ImageDraw
import streamlit as st
from classifier.ModelClassifier import predict

def main():
    st.title("Image Classifier")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image file
        image = Image.open(uploaded_file)
        label_predicted=predict(image)
        st.write(f"predict: {label_predicted}")
        st.image(image, caption="Uploaded Image", use_column_width=True)


if __name__ == "__main__":
    main()

