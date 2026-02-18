import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from timm import create_model

# --------------------------------
# Model Definition (EXACT MATCH)
# --------------------------------
class EfficientNetV2(nn.Module):
    def __init__(self, num_classes=1, dropout_rate=0.3, pretrained=False):
        super().__init__()
        self.base_model = create_model(
            "tf_efficientnetv2_l",
            pretrained=pretrained,
            num_classes=0
        )

        num_features = self.base_model.num_features

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(num_features, 512),
            nn.SiLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.base_model.forward_features(x)
        out = self.classifier(features)
        return out.squeeze(1)


# --------------------------------
# Config
# --------------------------------
MODEL_PATH = "saved_models/final_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ✅ Modern thresholds (tuned for old models)
REAL_THRESHOLD = 0.25
FAKE_THRESHOLD = 0.40


# --------------------------------
# Load Model (cached)
# --------------------------------
@st.cache_resource
def load_model():
    model = EfficientNetV2()
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()


# --------------------------------
# Image Transform (MUST MATCH TRAINING)
# --------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    ),
])


# --------------------------------
# Streamlit UI
# --------------------------------
st.set_page_config(
    page_title="Deepfake Image Detector",
    layout="centered"
)

st.title("🕵️ Deepfake Image Detection")
st.write(
    "This detector is calibrated for **modern deepfakes** using adjusted thresholds.\n\n"
    "**Outputs:** Real · Suspicious · Fake"
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logit = model(img_tensor)
        prob = torch.sigmoid(logit).item()

    # --------------------------------
    # Threshold-based Decision Logic
    # --------------------------------
    if prob >= FAKE_THRESHOLD:
        prediction = "Fake"
        confidence = prob
        st.error("⚠️ Likely Deepfake")

    elif prob <= REAL_THRESHOLD:
        prediction = "Real"
        confidence = 1 - prob
        st.success("✅ Likely Real")

    else:
        prediction = "Suspicious"
        confidence = 0.5
        st.warning("⚠️ Uncertain / Suspicious")

    st.markdown("---")
    st.subheader(f"Prediction: **{prediction}**")
    st.write(f"Confidence: **{confidence * 100:.2f}%**")

    # Debug info (optional but useful)
    with st.expander("🔍 Model Score (debug)"):
        st.write(f"Sigmoid probability: `{prob:.4f}`")
