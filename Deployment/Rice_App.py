import streamlit as st
import PIL
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import pandas as pd

# Weights file of the trained model
model_path = 'E:\\Streamlit\yolov8.pt'


def check_hashes(password, hashed_text):
    return hashed_text if make_hashes(password) == hashed_text else False
# Setting page layout
st.set_page_config(
    page_title="Rice Classification",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Function to load the model
def load_model(path):
    try:
        return YOLO(path)
    except Exception as ex:
        st.error(f"Unable to load model. Check the specified path: {model_path}")
        st.error(ex)
        return None

# Function to perform object detection
def detect_objects(model, image, conf):
    return model.predict(image, conf=conf)

# Function to plot bar chart
def plot_bar_chart(classes, counts):
    fig, ax = plt.subplots()
    ax.bar(classes, counts, color='green')
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    ax.set_title('Rice Class Distribution')
    return fig

# Function to plot pie chart
def plot_pie_chart(classes, counts):
    fig, ax = plt.subplots()
    ax.pie(counts, labels=classes, autopct='%1.1f%%', startangle=140)
    ax.axis('equal')
    return fig

# Creating sidebar
with st.sidebar:
    st.header("Image Config")
    source_img = st.file_uploader("Upload an image...", type=("jpg", "jpeg"))
    confidence = float(st.slider("Select Model Confidence", 25, 100, 40)) / 100

# Creating main page heading
st.title("Rice Detection")
col1, col2 = st.columns(2)

# Adding image to the first column if image is uploaded
if source_img:
    uploaded_image = PIL.Image.open(source_img)
    col1.image(source_img, caption="Uploaded Image", use_column_width=True)

model = load_model(model_path)

if model and st.sidebar.button('Detect Objects'):
    res = detect_objects(model, uploaded_image, confidence)
    boxes = res[0].boxes
    res_plotted = res[0].plot()[:, :, ::-1]
    col2.image(res_plotted, caption='Detected Image', use_column_width=True)

    if boxes:
        with st.expander("Detection Results", expanded=True):
            # Create an empty dictionary to store the counts per class
            class_counts = {}
            class_confidences = {}
            for box in boxes:
                label = box.cls.item()
                confidence_score = box.conf.item()
                if label not in class_counts:
                    class_counts[label] = 1
                    class_confidences[label] = [confidence_score]
                else:
                    class_counts[label] += 1
                    class_confidences[label].append(confidence_score)

            # Calculate average confidence per class
            avg_confidences = {label: np.mean(conf) for label, conf in class_confidences.items()}

            # Create a DataFrame to display the results
            data = {
                "Class": [model.names[label] for label in class_counts.keys()],
                "Count": [count for count in class_counts.values()],
                "Avg Confidence (%)": [f"{avg_confidences[label] * 100:.2f}" for label in class_counts.keys()]
            }
            df = pd.DataFrame(data)
            max_class = df["Count"].idxmax()

            # Function to apply styling to the DataFrame
            def highlight_max_row(s):
                return ['background-color: lightgreen; font-weight: bold'] * len(s) if s.name == max_class else [''] * len(s)

            # Apply the styling
            df_styled = df.style.apply(highlight_max_row, axis=1)
            df_styled = df_styled.set_properties(subset=['Class', 'Count', 'Avg Confidence (%)'], **{'text-align': 'left'})
            df_styled = df_styled.set_table_styles([dict(selector='th', props=[('text-align', 'left')])])

            # Display the total objects detected
            st.markdown(f"**Total Rice Grains Detected: {len(boxes)}**")

            # Display the styled DataFrame
            st.table(df_styled)

            # Plot bar chart and pie chart side by side
            st.subheader("Class Distribution")
            col3, col4 = st.columns(2)
            with col3:
                st.pyplot(plot_bar_chart(df["Class"], df["Count"]))
            with col4:
                st.pyplot(plot_pie_chart(df["Class"], df["Count"]))
