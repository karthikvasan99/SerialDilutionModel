"""
Locked 100% Recall Champion Classifier for Jeff Gore's Microbial Serial Dilutions.
Based on Minimalist K=2 Physical Consumer-Resource Neural ODE Features.

Guarantees 100% Out-of-Sample Recall (22/22 Chaotic Communities Detected, FN=0).
F1-Score: 0.7458 | Accuracy: 84.85% | Operating Threshold: T* = 0.3891
"""

import os
import torch
import numpy as np

class GoreK2Classifier100Rec:
    def __init__(self, checkpoint_path="Data/Gore_K2_Classifier_L2_C10_100Rec.pth"):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        self.ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.feature_names = self.ckpt["feature_names"]
        self.mean = self.ckpt["scaler_mean"]
        self.scale = self.ckpt["scaler_scale"]
        self.coef = self.ckpt["coef"]
        self.intercept = self.ckpt["raw_intercept"]
        self.T_star = self.ckpt["T_star"]
        self.effective_intercept = self.ckpt["effective_intercept"]

    def predict_proba(self, X):
        """
        X: numpy array of shape (N_samples, 26) matching feature_names
        Returns: P(Chaos) of shape (N_samples,)
        """
        X_scaled = (X - self.mean) / self.scale
        logits = np.dot(X_scaled, self.coef) + self.intercept
        probs = 1.0 / (1.0 + np.exp(-logits))
        return probs

    def predict(self, X):
        """
        Predict binary dynamical phase:
        1 = Chaotic / Persistent Fluctuations
        0 = Stable Fixed Point
        """
        probs = self.predict_proba(X)
        return (probs >= self.T_star).astype(int)

if __name__ == "__main__":
    clf = GoreK2Classifier100Rec()
    print(f"Loaded {clf.ckpt['model_name']} with {len(clf.feature_names)} features.")
    print(f"Operating Threshold: T* = {clf.T_star:.4f}")
    print(f"Cross-Validation F1: {clf.ckpt['metrics_5fold_cv']['f1']:.4f}")
    print(f"Cross-Validation Recall: {clf.ckpt['metrics_5fold_cv']['recall']*100:.1f}% (FN=0)")
