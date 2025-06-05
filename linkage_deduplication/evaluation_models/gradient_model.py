"""
gradient boosted tree comparison model
"""

import torch
import numpy as np
from .gbt import GradientBoostedClassifier


class GradientModel:
    """gradient boosted tree model for subject comparison"""

    def __init__(
        self, n_estimators=50, learning_rate=0.1, max_depth=3, device=None, verbose=True
    ):
        self.device = self._setup_device(device, verbose)
        self.classifier = GradientBoostedClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=2,
            min_samples_leaf=1,
            device=self.device,
            verbose=verbose,
        )
        self.features = [
            "first_name",
            "middle_name",
            "last_name",
            "dob",
            "dod",
            "email",
            "birth_city",
        ]
        self.is_trained = False

    def _setup_device(self, device, verbose=True):
        """setup and validate device selection"""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if isinstance(device, str):
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            device = torch.device(device)

        if device.type == "cuda" and not torch.cuda.is_available():
            if verbose:
                print("cuda requested but not available, falling back to cpu")
            device = torch.device("cpu")

        if verbose:
            print(f"GradientModel using device: {device}")
            if device.type == "cuda":
                print(f"cuda device: {torch.cuda.get_device_name(device)}")

        return device

    def to(self, device):
        """move model to specified device"""
        self.device = self._setup_device(device, False)
        self.classifier.to(self.device)
        return self

    def cuda(self):
        """move model to cuda device"""
        return self.to("cuda")

    def cpu(self):
        """move model to cpu device"""
        return self.to("cpu")

    def _extract_features(self, subject1, subject2):
        """extract comparison features between two subjects"""
        features = []
        for feature in self.features:
            attr1 = getattr(subject1, feature, "")
            attr2 = getattr(subject2, feature, "")
            features.append(1.0 if attr1 == attr2 else 0.0)
        return np.array(features)

    def gradient_boosted_score(self, subject1, subject2):
        """
        calculate similarity score between two subjects

        returns float similarity score 0->1
        """
        if not self.is_trained:
            # fallback to simple weighted scoring
            features = self._extract_features(subject1, subject2)
            weights = np.array([1.0, 0.8, 1.2, 1.5, 1.3, 1.1, 0.9])

            score = np.dot(weights, features) / np.sum(weights)
            return float(score)

        features = self._extract_features(subject1, subject2)
        features = features.reshape(1, -1)
        # move features to the correct device
        features_tensor = torch.tensor(
            features, dtype=torch.float32, device=self.device
        )
        probability = self.classifier.predict_proba(features_tensor)
        return float(probability[0])

    def train_gbt(self, subject_pairs, labels, epochs=10, lr=0.1, device="cuda"):
        """
        train the gradient boosted model

        args:
            subject_pairs: list of (subject1, subject2) tuples
            labels: list of binary labels (0/1)
            epochs: number of training epochs
            lr: learning rate (for gradient boosting)
            device: device to train on
        """
        # set device and learning rate
        self.to(device)
        self.classifier.learning_rate = lr

        # extract features from subject pairs
        X = []
        for subject1, subject2 in subject_pairs:
            features = self._extract_features(subject1, subject2)
            X.append(features)

        X = np.array(X)
        y = np.array(labels)

        # train the classifier
        self.classifier.train_gbt(X, y, epochs=epochs)
        self.is_trained = True

    def fit(self, subject_pairs, labels):
        """backward compatibility fit method"""
        self.train_gbt(subject_pairs, labels, epochs=1)
        return self

    def predict(self, subject_pairs):
        """predict labels for subject pairs"""
        scores = []
        for subject1, subject2 in subject_pairs:
            score = self.gradient_boosted_score(subject1, subject2)
            scores.append(score)

        return (np.array(scores) > 0.5).astype(int)

    def predict_proba(self, subject_pairs):
        """predict probabilities for subject pairs"""
        scores = []
        for subject1, subject2 in subject_pairs:
            score = self.gradient_boosted_score(subject1, subject2)
            scores.append(score)
        return np.array(scores)

    def save(self, path):
        """save the trained model"""
        if self.is_trained:
            self.classifier.save(path)
        else:
            raise ValueError("model must be trained before saving")

    def load(self, path):
        """load pre-trained model"""
        self.classifier.load(path)
        self.is_trained = True
