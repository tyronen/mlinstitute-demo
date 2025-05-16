import asyncio

asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

import streamlit as st
import torch
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from torchvision import transforms
import random
import string
from db import log_prediction, get_all_predictions
from models import MnistCNN, MODEL_PATH


@st.cache_resource
def load_model():
    model = MnistCNN()
    model_dict = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
    model.load_state_dict(model_dict["model_state_dict"])
    model.eval()
    mean = model_dict["mean"]
    std = model_dict["std"]
    return model, mean, std


def preprocess_image(image_data, mean, std):
    transform = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((mean,), (std,)),
        ]
    )
    image = Image.fromarray(image_data)
    return transform(image).unsqueeze(0)


def random_string():
    return "".join(
        random.choice(string.ascii_letters + string.digits) for _ in range(8)
    )


TEMPERATURE = 2.0

INTRO = """
This is a demonstration app showing simple handwriting recognition.

This app uses a deep-learning model trained on the MNIST public dataset of 
handwritten digits using the Pytorch library.

Use your mouse to draw a digit (0-9) in the black box and press Predict. The 
model will then attempt to guess what digit you have entered, and how confident
it is in that guess as a percentage. You can then press Submit to add each
guess to the prediction history below."""


def main():
    if "prediction" not in st.session_state:
        st.session_state.prediction = None
    if "confidence" not in st.session_state:
        st.session_state.confidence = None
    if "has_prediction" not in st.session_state:
        st.session_state.has_prediction = False
    if "canvas_key" not in st.session_state:
        st.session_state.canvas_key = random_string()

    st.title("Digit Recogniser")
    st.markdown(INTRO)

    canvas = st_canvas(
        fill_color="black",
        stroke_width=20,
        stroke_color="white",
        background_color="black",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key=st.session_state.canvas_key,
    )

    model, mean, std = load_model()

    if st.button("Predict") and canvas.image_data is not None:
        image_tensor = preprocess_image(canvas.image_data, mean, std)
        with torch.no_grad():
            outputs = model(image_tensor)
            scaled_outputs = outputs / TEMPERATURE
            probabilities = torch.nn.functional.softmax(scaled_outputs, dim=1)[0]
            prediction = torch.argmax(probabilities).item()
            confidence = probabilities[prediction].item() * 100

        st.session_state.prediction = prediction
        st.session_state.confidence = confidence
        st.session_state.has_prediction = True

    if st.session_state.has_prediction:
        st.write(f"**Prediction:** {st.session_state.prediction}")
        st.write(f"**Confidence:** {st.session_state.confidence:.1f}%")

        true_label = st.number_input(
            "True label:", min_value=0, max_value=9, step=1, value=None
        )

        if st.button("Submit") and true_label is not None:
            log_prediction(
                st.session_state.prediction, st.session_state.confidence, true_label
            )
            st.session_state.has_prediction = False
            get_all_predictions.clear()
            st.session_state.canvas_key = random_string()
            st.rerun()

    predictions_df = get_all_predictions()
    if not predictions_df.empty:
        st.write("### Prediction History")
        st.dataframe(
            predictions_df,
            hide_index=True,
            column_config={
                "True Label": st.column_config.NumberColumn(
                    "True Label",
                    width="small",
                ),
                "Confidence": st.column_config.NumberColumn(
                    "Confidence",
                    format="%.1f",
                    width="small",
                ),
                "Prediction": st.column_config.NumberColumn(
                    "Prediction",
                    width="small",
                ),
                "Timestamp": st.column_config.DatetimeColumn(
                    "Timestamp",
                    width="small",
                ),
            },
        )
    else:
        st.write("No predictions logged yet.")


if __name__ == "__main__":
    main()
