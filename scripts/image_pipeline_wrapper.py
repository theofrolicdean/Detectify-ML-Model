import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from sklearn.base import BaseEstimator, TransformerMixin
from timm import create_model

class ImagePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        self.transform_pipeline = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        X: list of file paths or list of PIL Images
        """
        print(f"DEBUG: preprocessor.transform type(X)={type(X)} len(X)={len(X)}", flush=True)
        transformed = []
        for item in X:
            if isinstance(item, str):
                image = Image.open(item)
            else:
                image = item
            
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            transformed.append(self.transform_logic(image))
        
        res = torch.stack(transformed)
        print(f"DEBUG: preprocessor.transform returning type={type(res)} shape={res.shape}", flush=True)
        return res

    def transform_logic(self, image):
        # Rename internal transform to transform_logic to avoid confusion with Scikit-learn transform
        return self.transform_pipeline(image)

class EfficientNetV2(nn.Module):
    def __init__(self, num_classes=1, dropout_rate=0.3, pretrained=False):
        super().__init__()
        self.base_model = create_model("tf_efficientnetv2_l", pretrained=pretrained, num_classes=0)
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
        if out.dim() == 2 and out.size(1) == 1:
            out = out.squeeze(1)
        return out

class ImageEfficientNetWrapper(BaseEstimator):
    def __init__(self, model_path=None, model=None, device="cpu"):
        self.model_path = model_path
        self.model = model
        self.device = device
        
    def fit(self, X, y=None):
        return self
    
    def predict(self, X):
        print(f"DEBUG: classifier.predict type(X)={type(X)}", flush=True)
        self._check_model()
        self.model.eval()
        with torch.no_grad():
            # If X is already a tensor (from preprocessor), use it.
            # Scikit-learn might wrap it if not careful.
            if isinstance(X, list):
                # This should not happen if Preprocessor returned a tensor
                X = torch.stack(X)
            
            X = X.to(self.device)
            outputs = self.model(X)
            preds = (torch.sigmoid(outputs) > 0.5).long()
        return preds.cpu().numpy()

    def predict_proba(self, X):
        self._check_model()
        self.model.eval()
        with torch.no_grad():
            X = X.to(self.device)
            outputs = self.model(X)
            probs = torch.sigmoid(outputs).unsqueeze(1)
            # Binary classification: [prob_class_0, prob_class_1]
            # Here outputs is prob_class_1 (fake)
            # Actually sigmoid(outputs) is P(fake)
            prob_fake = probs.cpu().numpy()
            prob_real = 1 - prob_fake
            return np.hstack([prob_real, prob_fake])

    def _check_model(self):
        if self.model is None:
            if self.model_path and os.path.exists(self.model_path):
                self.model = EfficientNetV2().to(self.device)
                checkpoint = torch.load(self.model_path, map_location=self.device)
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    self.model.load_state_dict(checkpoint)
            else:
                raise ValueError(f"Model not loaded and model_path {self.model_path} not found.")
import numpy as np
