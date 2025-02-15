import streamlit as st
import supervision as sv
from inference_sdk import InferenceHTTPClient
from collections import Counter
from PIL import Image, ImageDraw, ImageFont
import io
import numpy as np
import cv2

# Initialize the InferenceHTTPClient
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="mA2lEBUxzg7jOnRLGciO"
)

def process_image(image):
    # Perform inference
    result = CLIENT.infer(image, model_id="printed-circuit-board/3")

    print(result)

    # Extract the 'predictions' list from the result
    predictions = result.get('predictions', [])

    # Count the occurrences of each class
    class_counts = Counter(pred['class'] for pred in predictions)

    return class_counts, result

def draw_bounding_boxes(image, predictions):
    # load the results into the supervision Detections api
    detections = sv.Detections.from_inference(predictions)

    # create supervision annotators
    bounding_box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    # annotate the image with our inference results
    annotated_image = bounding_box_annotator.annotate(
        scene=image, detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections)

    return annotated_image

# Streamlit app
st.title("PCB Object Detection App")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Process the image and get results
    if st.button("Detect Objects"):
        with st.spinner("Detecting objects..."):
            class_counts, result = process_image(image)

        # Display results
        st.subheader("Detection Results")
        st.write(f"Number of classes detected: {len(class_counts)}")
        
        # Display results in a table format
        st.write("Objects detected per class:")
        results_table = [{"Class": class_name, "Count": count} for class_name, count in class_counts.items()]
        st.table(results_table)

        # Draw bounding boxes on the image
        annotated_image = draw_bounding_boxes(image.copy(), result)

        # Display the annotated image
        st.subheader("Annotated Image")
        st.image(annotated_image, caption="Annotated Image", use_container_width=True)

