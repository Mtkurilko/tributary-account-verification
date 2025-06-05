"""
gradient boosted tree comparison model
"""

import torch
import numpy as np
from .gbt import GradientBoostedClassifier


class GradientModel:
    """gradient boosted tree model for subject comparison"""

    def __init__(self, n_estimators=50, learning_rate=0.1, max_depth=3):
        self.classifier = GradientBoostedClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=2,
            min_samples_leaf=1,
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

        returns fload similarity score 0->1
        """
        if not self.is_trained:
            # fallback to simple weighted scoring
            features = self._extract_features(subject1, subject2)
            weights = np.array([1.0, 0.8, 1.2, 1.5, 1.3, 1.1, 0.9])

            score = np.dot(weights, features) / np.sum(weights)
            return float(score)

        features = self._extract_features(subject1, subject2)
        features = features.reshape(1, -1)
        probability = self.classifier.predict_proba(features)
        return float(probability[0])

    def fit(self, subject_pairs, labels):
        """training"""
        X = []
        for subject1, subject2 in subject_pairs:
            features = self._extract_features(subject1, subject2)
            X.append(features)

        X = np.array(X)
        y = np.array(labels)

        # train the classifier
        self.classifier.fit(X, y)
        self.is_trained = True

    def predict(self, subject_pairs):
        """predict labels for subject pairs"""
        scores = []
        for subject1, subject2 in subject_pairs:
            score = self.gradient_boosted_score(subject1, subject2)
            scores.append(score)

        return (np.array(scores) > 0.5).astype(int)

    def get_feature_importance(self):
        """get feature importance for trained model"""
        if not self.is_trained:
            return None
        return self.classifier.get_feature_importance()

    def save(self, path):
        """save the trained model"""
        if self.is_trained:
            self.classifier.save(path)
        else:
            raise ValueError("model must be trained before saving")

    def load(self, path):
        """load pre-trained"""
        self.classifier.load(path)
        self.is_trained = True
