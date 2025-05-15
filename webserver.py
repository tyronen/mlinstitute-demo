import asyncio

asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

# Rest of your imports
import streamlit as st
import torch
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from torchvision import transforms

from db import log_prediction, get_all_predictions
from models import MnistCNN, MODEL_PATH


@st.cache_resource
def load_model():
    model = MnistCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    return model


def preprocess_image(image_data):
    transform = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # MNIST normalization values
        ]
    )
    image = Image.fromarray(image_data)
    return transform(image).unsqueeze(0)


def main():
    if "prediction" not in st.session_state:
        st.session_state.prediction = None
    if "confidence" not in st.session_state:
        st.session_state.confidence = None
    if "has_prediction" not in st.session_state:
        st.session_state.has_prediction = False

    st.title("Digit Recogniser")
    canvas = st_canvas(
        fill_color="black",
        stroke_width=20,
        stroke_color="white",
        background_color="black",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    model = load_model()

    if st.button("Predict") and canvas.image_data is not None:
        image_tensor = preprocess_image(canvas.image_data)
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            prediction = torch.argmax(probabilities).item()
            confidence = probabilities[prediction].item() * 100

        st.session_state.prediction = prediction
        st.session_state.confidence = confidence
        st.session_state.has_prediction = True

    if st.session_state.has_prediction:
        st.write(f"**Prediction:** {st.session_state.prediction}")
        st.write(f"**Confidence:** {st.session_state.confidence:.2f}%")

        true_label = st.number_input(
            "True label:", min_value=0, max_value=9, step=1, value=None
        )

        if st.button("Submit") and true_label is not None:
            log_prediction(
                st.session_state.prediction, st.session_state.confidence, true_label
            )
            st.session_state.has_prediction = False
            get_all_predictions.clear()
            st.rerun()

    # Get prediction data - now returns a DataFrame with column headers
    predictions_df = get_all_predictions()
    if not predictions_df.empty:
        st.write("### Prediction History")
        st.dataframe(predictions_df, hide_index=True)
    else:
        st.write("No predictions logged yet.")


if __name__ == "__main__":
    main()
