import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights
from PIL import Image



st.set_page_config(page_title="🌿 Plant Disease Detection", layout="centered")


st.markdown(
    """
    <style>
    /* Page background */
    .stApp { background: linear-gradient(135deg,#f0fdf4 0%, #ecfeff 100%); }

    /* Card */
    .card { background: #fff; padding: 22px; border-radius: 16px; box-shadow: 0 18px 40px rgba(2,6,23,0.06); }

    /* Header */
    .app-header { font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif; font-weight:700; font-size:24px; }
    .app-sub { color: #64748b; margin-top: -6px; }

    /* Upload area */
    .upload-area { border: 2px dashed rgba(16,185,129,0.18); border-radius: 12px; padding: 18px; text-align:center; background: linear-gradient(180deg, rgba(16,185,129,0.03), rgba(99,102,241,0.01)); }
    .upload-area:hover{ box-shadow: 0 12px 30px rgba(16,185,129,0.08); transform: translateY(-2px); transition: all .18s ease }

    /* Buttons */
    .stButton>button { border-radius: 10px; padding: .6rem 1.2rem; font-weight:600 }
    .stButton>button.primary { background: linear-gradient(90deg,#10b981,#6366f1); border: none; color:white }

    /* Preview image */
    .stImage>div>img { border-radius: 12px; box-shadow: 0 12px 30px rgba(2,6,23,0.06); }

    /* Result badge */
    .result-badge { display:inline-block; padding:10px 16px; border-radius:999px; font-weight:700; background:#ecfdf5; color:#065f46 }

    /* Small helpers */
    .small-muted { color:#64748b; font-size:0.95rem }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='app-header'>🌿 Plant Disease Detection</div>", unsafe_allow_html=True)
st.markdown("<div class='app-sub'>Upload a plant leaf image to identify the disease or check if it's healthy.</div>", unsafe_allow_html=True)


# Class Labels

CLASS_NAMES = [
    "Pepper_bell_Bacterial_spot",
    "Pepper_bell_healthy",
    "Potato_Early_blight",
    "Potato_healthy",
    "Potato_Late_blight",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato_Target_Spot",
    "Tomato_Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato_Tomato_mosaic_virus",
    "Tomato_healthy"
]


# Load Model

@st.cache_resource
def load_model():
    try:
        # Load checkpoint first to detect how many output classes it was trained with.
        ckpt = torch.load("plant_disease_model.pth", map_location="cpu")

        # If the checkpoint is an nn.Module (full model), return it directly.
        if not isinstance(ckpt, dict):
            ckpt.eval()
            return ckpt

        # Try to infer saved output features from state_dict keys
        saved_out = None
        for k, v in ckpt.items():
            if k.endswith('classifier.1.weight') or k.endswith('classifier.1.1.weight'):
                saved_out = v.shape[0]
                break

        
        model = models.efficientnet_b0(weights=None)
        num_features = model.classifier[1].in_features

        target_out = len(CLASS_NAMES)
        load_out = saved_out or target_out

        
        model.classifier[1] = nn.Linear(num_features, load_out)

       
        try:
            model.load_state_dict(ckpt)
        except Exception:
            
            model.load_state_dict(ckpt, strict=False)

        
        if load_out != target_out:
            st.warning(f"Model checkpoint has {load_out} outputs but app expects {target_out} classes. Copying {min(load_out, target_out)} matching weights and randomly initializing the rest.")
            new_fc = nn.Linear(num_features, target_out)
            with torch.no_grad():
                # Copy overlapping weights/bias
                ncopy = min(load_out, target_out)
                new_fc.weight[:ncopy] = model.classifier[1].weight[:ncopy]
                new_fc.bias[:ncopy] = model.classifier[1].bias[:ncopy]
            model.classifier[1] = new_fc

        model.eval()
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

with st.spinner("Loading model... ⏳"):
    model = load_model()

if model is None:
    st.error("Model failed to load. Check terminal logs for details.")
    st.stop()  # Stop app if model failed to load

# -------------------------------
# Image Upload
# -------------------------------
# Use session state for the uploader so we can clear it reliably
st.file_uploader("", type=["jpg", "jpeg", "png"], help="Upload a clear photo of the leaf", key="uploaded_file")
uploaded_file = st.session_state.get("uploaded_file")

# Layout: nicer card with two columns
with st.container():
    left, right = st.columns([2, 3])

    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Upload & Predict", unsafe_allow_html=True)
        st.markdown("<div class='small-muted'>Choose an image. Supported: JPG, PNG.</div>", unsafe_allow_html=True)

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.markdown(f"**Selected:** {uploaded_file.name}")

            # Clear button to remove the uploaded file from session state
            if st.button("Clear", key="clear_btn"):
                st.session_state["uploaded_file"] = None
                st.experimental_rerun()

            # Show Predict button only when a file is present
            predict_clicked = st.button("Analyze Leaf 🌿", key="predict_btn")
        else:
            st.info("Please upload an image to enable analysis.")
            predict_clicked = False

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Preview & Result", unsafe_allow_html=True)

        if uploaded_file is None:
            st.info("👆 Upload an image to start analysis.")
        else:
            st.image(Image.open(uploaded_file), caption="Uploaded Image", use_container_width=True)

            if predict_clicked:
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
                ])
                img = transform(image).unsqueeze(0)

                with st.spinner("Running analysis... ⏳"):
                    with torch.no_grad():
                        outputs = model(img)
                        _, predicted = torch.max(outputs, 1)
                        class_name = CLASS_NAMES[predicted.item()]

                # styled result card
                st.markdown(
                    f"<div style='padding:12px; border-radius:12px; background:linear-gradient(90deg,#f0fff4,#f8fbff);'><div class='small-muted'>Model: EfficientNet-B0</div><h4 style='margin-top:8px'>✅ {class_name}</h4></div>",
                    unsafe_allow_html=True,
                )

                # show confidence if available
                try:
                    probs = torch.nn.functional.softmax(outputs, dim=1)[0]
                    conf = float(probs[predicted].item())
                    st.success(f"Confidence: {conf:.2%}")
                except Exception:
                    pass

        st.markdown("</div>", unsafe_allow_html=True)
