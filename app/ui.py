"""Interactive Streamlit UI for MNIST CNN training and visualization."""

from io import BytesIO
from typing import Tuple

import matplotlib.pyplot as plt
import streamlit as st
import torch
import torch.optim as optim
from PIL import Image, ImageOps

from gpu.gpu import get_device, get_device_info
from model.mnist_cnn import (
    MNISTNet,
    evaluate,
    get_dataloaders,
    get_transforms,
    predict_single,
    train_one_epoch,
)
from utils.plotting import (
    build_confusion_matrix_figure,
    build_feature_map_figure,
    build_filter_figure,
    build_predictions_grid,
    build_probability_bar_figure,
)


st.set_page_config(page_title="MNIST CNN Lab", page_icon="MN", layout="wide")

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=DM+Serif+Display:ital@0;1&display=swap');
            :root {
                --ink: #17263c;
                --muted-ink: #33435b;
                --panel: rgba(255, 255, 255, 0.8);
                --accent-1: #1d4e89;
                --accent-2: #d17b0f;
            }
      .stApp {
                background: radial-gradient(circle at 12% 18%, #f7f2e8 0%, #ece4d8 40%, #dbe6ef 100%);
                color: var(--ink);
      }
      h1, h2, h3 {
        font-family: 'DM Serif Display', serif;
                color: var(--ink);
        letter-spacing: 0.4px;
      }
            p, div, label, span, li {
        font-family: 'Space Grotesk', sans-serif;
                color: var(--ink);
      }
            .stMarkdown p, .stMarkdown li, .stMarkdown span,
            .stSelectbox label, .stRadio label, .stSlider label,
            .stFileUploader label, .stTextInput label,
            [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
                color: var(--ink) !important;
            }
            [data-testid="stSidebar"] {
                background: rgba(245, 242, 235, 0.92);
            }
            [data-testid="stSidebarContent"] {
                padding-top: 0.35rem !important;
            }
            [data-testid="stSidebarContent"] .block-container {
                padding-top: 0.2rem !important;
                padding-bottom: 0.4rem !important;
            }
            [data-testid="stSidebarContent"] [data-testid="stVerticalBlock"] {
                gap: 0.3rem !important;
            }
            div[data-baseweb="select"] > div,
            .stTextInput input,
            .stNumberInput input,
            .stTextArea textarea {
                background: var(--panel);
                color: var(--ink);
                border-color: rgba(23, 38, 60, 0.25);
            }
      .hero {
                border: 1px solid rgba(29, 78, 137, 0.35);
        border-radius: 18px;
                background: linear-gradient(110deg, rgba(248, 242, 230, 0.92), rgba(225, 236, 247, 0.92));
        padding: 1.2rem 1.3rem;
                box-shadow: 0 10px 20px rgba(23, 38, 60, 0.1);
      }
      .metric-card {
                border: 1px solid rgba(23, 38, 60, 0.24);
        border-radius: 14px;
                background: rgba(230, 240, 255, 0.85);
        padding: 0.8rem;
      }
      [data-testid="stSidebar"] .stButton > button {
                background: rgba(230, 240, 255, 0.85) !important;
                color: var(--ink) !important;
                border: 1px solid rgba(23, 38, 60, 0.24) !important;
                border-radius: 8px !important;
      }
      [data-testid="stSidebar"] .stButton > button:hover,
      [data-testid="stSidebar"] .stButton > button:focus {
                background: rgba(200, 220, 255, 0.90) !important;
                border-color: rgba(23, 38, 60, 0.35) !important;
      }
            .stAlert {
                background: rgba(255, 255, 255, 0.9);
            }
            .stButton > button,
            button[kind="secondary"] {
                background: rgba(230, 240, 255, 0.85) !important;
                color: var(--ink) !important;
                border: 1px solid rgba(23, 38, 60, 0.24) !important;
            }
            button[kind="primary"] {
                background: rgba(230, 240, 255, 0.85) !important;
                color: var(--ink) !important;
                border: 1px solid rgba(23, 38, 60, 0.24) !important;
            }
            .stButton > button:hover,
            .stButton > button:focus,
            button[kind="secondary"]:hover,
            button[kind="secondary"]:focus {
                background: rgba(200, 220, 255, 0.90) !important;
                color: var(--ink) !important;
                border-color: rgba(23, 38, 60, 0.35) !important;
                box-shadow: none !important;
            }
            button[kind="primary"]:hover,
            button[kind="primary"]:focus {
                background: rgba(29, 78, 137, 0.08) !important;
                color: var(--ink) !important;
                border-color: var(--accent-1) !important;
                box-shadow: none !important;
            }
            .stFileUploader,
            [data-testid="stFileUploader"],
            [data-testid="stFileUploader"] > div {
                background: rgba(255, 255, 255, 0.95) !important;
                color: #111111 !important;
                border-radius: 10px !important;
            }
            [data-testid="stFileUploader"] section,
            [data-testid="stFileUploader"] section > div,
            [data-testid="stFileUploader"] section > div > div {
                background: rgba(255, 255, 255, 0.95) !important;
                color: #111111 !important;
            }
            section[data-testid="stFileUploaderDropzone"] {
                background: rgba(255, 255, 255, 0.98) !important;
                border: 1px dashed rgba(23, 38, 60, 0.4) !important;
                color: #111111 !important;
            }
            section[data-testid="stFileUploaderDropzone"] * {
                background: transparent !important;
                color: #111111 !important;
            }
            section[data-testid="stFileUploaderDropzone"] button,
            section[data-testid="stFileUploaderDropzone"] button:hover,
            section[data-testid="stFileUploaderDropzone"] button:focus,
            [data-testid="stFileUploader"] button,
            [data-testid="stFileUploader"] button:hover,
            [data-testid="stFileUploader"] button:focus {
                background: rgba(230, 240, 255, 0.85) !important;
                color: #111111 !important;
                border: 1px solid rgba(23, 38, 60, 0.24) !important;
                box-shadow: none !important;
                padding: 8px 16px !important;
                border-radius: 6px !important;
                cursor: pointer !important;
            }
            [data-testid="stFileUploader"] button:hover,
            [data-testid="stFileUploader"] button:focus {
                background: rgba(200, 220, 255, 0.90) !important;
                border-color: rgba(23, 38, 60, 0.35) !important;
            }
            section[data-testid="stFileUploaderDropzone"] button span,
            section[data-testid="stFileUploaderDropzone"] button span *,
            [data-testid="stFileUploader"] button span,
            [data-testid="stFileUploader"] button span * {
                color: #111111 !important;
            }
            section[data-testid="stFileUploaderDropzone"] button svg,
            section[data-testid="stFileUploaderDropzone"] button path,
            [data-testid="stFileUploader"] button svg,
            [data-testid="stFileUploader"] button path {
                fill: #111111 !important;
                color: #111111 !important;
            }
            [data-testid="stFileUploaderFile"],
            [data-testid="stFileUploaderFile"] > div,
            [data-testid="stFileUploaderFile"] > div > div {
                display: none !important;
            }
            [data-testid="stFileUploaderFile"] *,
            [data-testid="stFileUploaderFile"] small,
            [data-testid="stFileUploaderFile"] span,
            [data-testid="stFileUploaderFile"] div,
            [data-testid="stFileUploaderFile"] p {
                display: none !important;
            }
            [data-testid="stFileUploaderFile"] svg,
            [data-testid="stFileUploaderFile"] path {
                display: none !important;
            }
            [data-testid="stFileUploader"] button[type="button"],
            [data-testid="stFileUploader"] button[type="button"]:hover,
            [data-testid="stFileUploader"] button[type="button"]:focus,
            [data-testid="stFileUploader"] button[type="button"]:active,
            .stFileUploader button,
            .stFileUploader button:hover,
            .stFileUploader button:focus,
            .stFileUploader button:active {
                background: rgba(255, 255, 255, 0.98) !important;
                color: #111111 !important;
                border: 1px solid rgba(23, 38, 60, 0.45) !important;
                box-shadow: none !important;
            }
            [data-testid="stFileUploader"] button[aria-label*="Remove"],
            [data-testid="stFileUploader"] button[title*="Remove"] {
                width: 2rem !important;
                height: 2rem !important;
                min-width: 2rem !important;
                min-height: 2rem !important;
                padding: 0 !important;
                border-radius: 999px !important;
                font-size: 0 !important;
                line-height: 0 !important;
                position: relative !important;
            }
            [data-testid="stFileUploader"] button[aria-label*="Remove"] svg,
            [data-testid="stFileUploader"] button[title*="Remove"] svg {
                display: none !important;
            }
            [data-testid="stFileUploader"] button[aria-label*="Remove"]::before,
            [data-testid="stFileUploader"] button[title*="Remove"]::before {
                content: "X";
                color: #111111;
                font-size: 1.15rem;
                line-height: 2rem;
                font-weight: 700;
                position: absolute;
                inset: 0;
                text-align: center;
            }
            [data-baseweb="tab-list"] {
                border-bottom: 2px solid rgba(29, 78, 137, 0.2);
            }
            button[role="tab"] {
                color: var(--muted-ink) !important;
            }
            button[aria-selected="true"][role="tab"] {
                color: var(--accent-1) !important;
                border-bottom: 3px solid var(--accent-2) !important;
            }
            details[data-testid="stExpander"] {
                background: transparent !important;
                border: 1px solid rgba(23, 38, 60, 0.3) !important;
            }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def cached_data(batch_size: int, limit_train: int, limit_test: int):
    return get_dataloaders(
        batch_size=batch_size,
        test_batch_size=1000,
        limit_train=limit_train,
        limit_test=limit_test,
    )


def load_uploaded_image(uploaded_file) -> torch.Tensor:
    image = Image.open(BytesIO(uploaded_file.read())).convert("L")
    image = ImageOps.invert(image)
    image = image.resize((28, 28))
    tensor = get_transforms()(image).unsqueeze(0)
    return tensor


def get_or_create_model(device: torch.device) -> Tuple[MNISTNet, optim.Optimizer]:
    if "model" not in st.session_state:
        st.session_state.model = MNISTNet().to(device)
    if "optimizer" not in st.session_state:
        st.session_state.optimizer = optim.Adam(st.session_state.model.parameters(), lr=0.001)
    if "history" not in st.session_state:
        st.session_state.history = []
    return st.session_state.model, st.session_state.optimizer


device = get_device()
device_info = get_device_info()
model, optimizer = get_or_create_model(device)

with st.sidebar:
    st.subheader("Setup")
    st.markdown(
        f"""
        <div class="metric-card">
          <p><b>Device:</b> {device_info['device']}</p>
          <p><b>GPU:</b> {device_info['gpu_name']}</p>
          <p><b>CUDA:</b> {device_info['cuda_version']}</p>
          <p><b>Torch:</b> {device_info['torch_version']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    st.subheader("View Mode")
    view_mode = st.radio("Select", ["1) Train", "2) Explore", "3) Predict"], label_visibility="collapsed")

    st.divider()
    if st.button("Reset Model", use_container_width=True):
        st.session_state.model = MNISTNet().to(device)
        st.session_state.optimizer = optim.Adam(st.session_state.model.parameters(), lr=0.001)
        st.session_state.history = []
        st.session_state.latest_eval = None
        st.session_state.training_completed = False
        st.success("Model reset complete.")
    
    st.divider()
    if view_mode == "1) Train":
        st.subheader("Show result")
        st.radio(
            "Select",
            ["Prediction Grid", "Confusion Matrix"],
            horizontal=True,
            key="train_result_view",
            label_visibility="collapsed",
        )

st.title("MNIST CNN Lab")
st.markdown(
    """
    Inspect what each layer of a trained convolutional neural network learns and interactively test your own handwritten digits.
    """
)

# Load default data for all views
default_batch_size = 64
default_limit_train = 6000
default_limit_test = 2000
default_train_loader, default_test_loader = cached_data(default_batch_size, default_limit_train, default_limit_test)

# Initialize session state for train loaders if not present
if "train_test_loader" not in st.session_state:
    st.session_state.train_test_loader = default_test_loader

# Keep a fixed batch for Explore activation maps to avoid jitter across reruns
if "explore_sample_images" not in st.session_state:
    explore_images, _ = next(iter(default_test_loader))
    st.session_state.explore_sample_images = explore_images

# Clear stale predict outputs before rendering right column
if view_mode == "3) Predict":
    current_source = st.session_state.get("predict_source_type", "Upload Your Digit")
    uploaded_state = st.session_state.get("predict_upload_file")
    if current_source == "Upload Your Digit" and uploaded_state is None:
        st.session_state.predict_tensor = None
        st.session_state.predict_true_label = None

main_col, right_col = st.columns([2, 2], gap="large")

# Right column: Output visualizations
with right_col:
    
    if view_mode == "1) Train":
        if st.session_state.history:
            latest = st.session_state.history[-1]
            if st.session_state.get("latest_eval") is None:
                st.session_state.latest_eval = evaluate(model, device, st.session_state.train_test_loader)
            
            eval_stats = st.session_state.latest_eval
            sample_images, sample_labels, sample_preds = eval_stats["sample_batch"]
            train_view_mode = st.session_state.get("train_result_view", "Prediction Grid")
            if train_view_mode == "Prediction Grid":
                st.pyplot(build_predictions_grid(sample_images, sample_labels, sample_preds))
            else:
                st.pyplot(build_confusion_matrix_figure(eval_stats["preds"], eval_stats["targets"]))
    
    elif view_mode == "2) Explore":
        explore_view_mode = st.session_state.get("explore_result_view", "Filters")
        
        if explore_view_mode == "Filters":
            st.pyplot(build_filter_figure(model))
        else:
            cached_png = st.session_state.get("explore_feature_png")
            if cached_png is not None:
                st.image(cached_png, width=640)
            else:
                st.info("Choose a sample/layer and click Generate Activation Maps.")
    
    elif view_mode == "3) Predict":
        input_tensor_predict = st.session_state.get("predict_tensor", None)
        true_label_predict = st.session_state.get("predict_true_label", None)

        if input_tensor_predict is not None:
            output = predict_single(model, input_tensor_predict, device)
            center_left, center_mid, center_right = st.columns([1, 6, 1])
            with center_mid:
                st.pyplot(build_probability_bar_figure(output["probabilities"], true_label=true_label_predict))
                pred_digit = int(output["prediction"].item())
                st.metric("Predicted", pred_digit)

# Main content area
with main_col:
    # ===== TRAIN VIEW =====
    if view_mode == "1) Train":
        st.subheader("Train and Evaluate")
        st.caption("Run training first, then inspect model quality below.")
        
        # Training controls
        batch_size = st.slider("Batch size", 32, 256, 64, step=32)
        limit_train = st.slider("Train samples", 500, 60000, 6000, step=500)
        limit_test = st.slider("Test samples", 500, 10000, 2000, step=500)
        epoch_count = st.slider("Epochs per run", 1, 8, 2)
        
        # Load data loaders
        train_loader, test_loader = cached_data(batch_size, limit_train, limit_test)
        
        # Store test_loader in session state for use in right column
        st.session_state.train_test_loader = test_loader
        
        # Initialize training completion flag
        if "training_completed" not in st.session_state:
            st.session_state.training_completed = False
        
        start = False
        if st.session_state.training_completed:
            st.success("✓ Training run finished.")
        else:
            start = st.button("Start Training", use_container_width=True)
        
        if start:
            progress = st.progress(0)
            for i in range(epoch_count):
                train_stats = train_one_epoch(model, device, train_loader, optimizer)
                eval_stats = evaluate(model, device, test_loader)
                st.session_state.latest_eval = eval_stats
                st.session_state.history.append(
                    {
                        "train_loss": train_stats["loss"],
                        "train_acc": train_stats["accuracy"],
                        "test_loss": float(eval_stats["loss"]),
                        "test_acc": float(eval_stats["accuracy"]),
                    }
                )
                progress.progress((i + 1) / epoch_count)
            st.session_state.training_completed = True
            st.rerun()
        
        if st.session_state.history:
            latest = st.session_state.history[-1]
            r1, r2 = st.columns(2)
            r1.metric("Train Loss", f"{latest['train_loss']:.4f}")
            r2.metric("Train Acc", f"{latest['train_acc'] * 100:.2f}%")
            r3, r4 = st.columns(2)
            r3.metric("Test Loss", f"{latest['test_loss']:.4f}")
            r4.metric("Test Acc", f"{latest['test_acc'] * 100:.2f}%")

    # ===== EXPLORE VIEW =====
    elif view_mode == "2) Explore":
        st.subheader("Explore What The CNN Learns")
        st.markdown(
            """
            Feature maps: brighter regions indicate stronger activations.
            
            **How to read this section**

            1. Conv1 filters usually capture edges and stroke directions.
            2. Conv2 combines simple strokes into digit parts.
            3. Dense layers convert these patterns into class scores.
            """
        )
        explore_view = st.radio(
            "Show view",
            ["Filters", "Activation Maps"],
            horizontal=True,
            key="explore_result_view",
        )
        
        if explore_view == "Activation Maps":
            sample_images_for_explore = st.session_state.explore_sample_images
            explore_idx = st.slider("Sample index", 0, sample_images_for_explore.shape[0] - 1, 0, key="explore_idx")
            layer_pick = st.radio("Layer", ["Conv1", "Conv2"], horizontal=True, key="explore_layer")
            if st.button("Generate Activation Maps", use_container_width=True):
                explore_input = sample_images_for_explore[explore_idx : explore_idx + 1]
                explore_output = predict_single(model, explore_input, device)
                if layer_pick == "Conv1":
                    feature_maps = explore_output["conv1"]
                    title = "Conv1 Activations"
                else:
                    feature_maps = explore_output["conv2"]
                    title = "Conv2 Activations"

                fig = build_feature_map_figure(feature_maps, title)
                buf = BytesIO()
                fig.savefig(buf, format="png", dpi=170, facecolor="white", bbox_inches="tight")
                plt.close(fig)
                st.session_state.explore_feature_png = buf.getvalue()
                st.rerun()

    # ===== PREDICT VIEW =====
    elif view_mode == "3) Predict":
        st.subheader("Predict A Digit")
        
        input_tensor = None
        true_label = None
        source_type = st.radio(
            "Select input",
            ["Upload Your Digit", "Random Test Sample"],
            horizontal=True,
            key="predict_source_type",
        )
        
        st.divider()
        
        if source_type == "Upload Your Digit":
            st.caption("Upload a handwritten digit (PNG/JPG/JPEG) for prediction")
            uploaded = st.file_uploader(
                "",
                type=["png", "jpg", "jpeg"],
                label_visibility="collapsed",
                key="predict_upload_file",
            )
            if uploaded is not None:
                st.caption(f"Uploaded file: {uploaded.name}")
                input_tensor = load_uploaded_image(uploaded)
            else:
                st.session_state.predict_tensor = None
                st.session_state.predict_true_label = None
                
        else:
            st.caption("Browse test set or jump to a specific sample")
            sample_batch = next(iter(default_test_loader))
            images, labels = sample_batch
            idx = st.slider("Sample", 0, images.shape[0] - 1, 0, label_visibility="collapsed")
            
            input_tensor = images[idx : idx + 1]
            true_label = int(labels[idx].item())
        
        predict_clicked = st.button("Predict", type="primary", use_container_width=True, disabled=input_tensor is None)

        # Store in session state for display in right column only after explicit action
        if predict_clicked and input_tensor is not None:
            st.session_state.predict_tensor = input_tensor
            st.session_state.predict_true_label = true_label
            st.rerun()
        elif input_tensor is None:
            st.session_state.predict_tensor = None
            st.session_state.predict_true_label = None

        # Show model input in the main section below the Predict button
        input_tensor_main = st.session_state.get("predict_tensor", None)
        true_label_main = st.session_state.get("predict_true_label", None)
        if input_tensor_main is not None:
            output_main = predict_single(model, input_tensor_main, device)
            pred_digit_main = int(output_main["prediction"].item())
            st.image(input_tensor_main[0].squeeze().numpy(), clamp=True, caption="Model Input", width=220)
            if true_label_main is not None:
                st.metric("True", true_label_main)