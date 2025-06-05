"""
gradient boosted tree implementation for pytorch
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional
import pickle


class TreeNode:
    """a node in a decision tree"""

    def __init__(
        self, feature_idx=None, threshold=None, left=None, right=None, value=None
    ):
        self.feature_idx = feature_idx  # feature index to split on
        self.threshold = threshold  # threshold for the split
        self.left = left  # left child node
        self.right = right  # right child node
        self.value = value  # leaf value for prediction

    def is_leaf(self):
        return self.value is not None


class DecisionTree:
    """decision tree for gradient boosting"""

    def __init__(self, max_depth=3, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root = None

    def fit(self, X, y):
        """fit the decision tree to the data"""
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        """recursively build the decision tree"""
        n_samples, n_features = X.shape

        # check stopping criteria
        if (
            depth >= self.max_depth
            or n_samples < self.min_samples_split
            or len(np.unique(y)) == 1
        ):
            # create leaf node with mean of y vals
            leaf_value = np.mean(y)
            return TreeNode(value=leaf_value)

        # find best split
        best_feature, best_threshold = self._find_best_split(X, y)

        if best_feature is None:
            # no good split found, create leaf
            leaf_value = np.mean(y)
            return TreeNode(value=leaf_value)

        # split data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        # check minimum samples per leaf
        if (
            np.sum(left_mask) < self.min_samples_leaf
            or np.sum(right_mask) < self.min_samples_leaf
        ):
            leaf_value = np.mean(y)
            return TreeNode(value=leaf_value)

        # recursively build left and right subtrees
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return TreeNode(
            feature_idx=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child,
        )

    def _find_best_split(self, X, y):
        """find the best feature and threshold to split on"""
        n_samples, n_features = X.shape
        best_gain = -float("inf")
        best_feature = None
        best_threshold = None

        current_mse = self._calculate_mse(y)

        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)

            for i in range(len(unique_values) - 1):
                threshold = (unique_values[i] + unique_values[i + 1]) / 2

                left_mask = feature_values <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                # calculate weighted mse after split
                left_mse = self._calculate_mse(y[left_mask])
                right_mse = self._calculate_mse(y[right_mask])

                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)

                weighted_mse = (n_left * left_mse + n_right * right_mse) / n_samples
                gain = current_mse - weighted_mse

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold

    def _calculate_mse(self, y):
        """calculate mean squared error"""
        if len(y) == 0:
            return 0
        mean_y = np.mean(y)
        return np.mean((y - mean_y) ** 2)

    def predict(self, X):
        """predict values for input samples"""
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        predictions = []
        for x in X:
            predictions.append(self._predict_single(x, self.root))
        return np.array(predictions)

    def _predict_single(self, x, node):
        """predict a single sample"""
        if node.is_leaf():
            return node.value

        if x[node.feature_idx] <= node.threshold:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)


class GradientBoostedTrees:
    """gradient boosted trees"""

    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
        device=None,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.trees = []
        self.initial_prediction = 0.0
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def fit(self, X, y):
        """fit the gradient boosted tree to the data"""
        # convert to tensors
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32, device=self.device)

        # initialize with mean of target values
        self.initial_prediction = torch.mean(y).item()
        predictions = torch.full((len(y),), self.initial_prediction, device=self.device)

        self.trees = []

        for i in range(self.n_estimators):
            # calculate residuals, negative gradients for mse loss
            residuals = y - predictions

            # fit tree to residuals
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
            )
            tree.fit(X, residuals)

            # update predictions
            tree_predictions = torch.tensor(tree.predict(X), device=self.device)
            predictions += self.learning_rate * tree_predictions

            self.trees.append(tree)

    def predict(self, X):
        """predict values for input samples"""
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)

        predictions = torch.full((len(X),), self.initial_prediction, device=self.device)

        for tree in self.trees:
            tree_predictions = torch.tensor(tree.predict(X), device=self.device)
            predictions += self.learning_rate * tree_predictions

        return predictions

    def predict_proba(self, X):
        """predict probabilities using sigmoid activation"""
        raw_predictions = self.predict(X)
        # apply sigmoid to convert to probabilities
        return torch.sigmoid(raw_predictions)


class GradientBoostedClassifier:
    """gradient boosted trees for binary classification"""

    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
        device=None,
    ):
        self.gb_trees = GradientBoostedTrees(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            device=device,
        )
        self.feature_importance = None

    def fit(self, X, y):
        """fit the classifier"""
        # convert to tensors if needed
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32)

        # convert binary labels to logits for training
        # y should be 0/1, convert to logits
        y_logits = torch.where(y == 1, torch.tensor(1.0), torch.tensor(-1.0))

        self.gb_trees.fit(X, y_logits)
        self._calculate_feature_importance(X)

    def _calculate_feature_importance(self, X):
        """calculate feature importance using pytorch"""
        n_features = X.shape[1]
        importance_scores = []

        for feature_idx in range(n_features):
            # count how often each feature is used for splitting
            feature_count = 0
            for tree in self.gb_trees.trees:
                feature_count += self._count_feature_usage(tree.root, feature_idx)
            importance_scores.append(feature_count)

        # normalize and convert to pytorch tensor
        importance_tensor = torch.tensor(importance_scores, dtype=torch.float32)
        if torch.sum(importance_tensor) > 0:
            importance_tensor = importance_tensor / torch.sum(importance_tensor)

        self.feature_importance = importance_tensor

    def _count_feature_usage(self, node, feature_idx):
        """recursively count usage times of a feature in the tree"""
        if node.is_leaf():
            return 0

        count = 1 if node.feature_idx == feature_idx else 0
        count += self._count_feature_usage(node.left, feature_idx)
        count += self._count_feature_usage(node.right, feature_idx)
        return count

    def predict_proba(self, X):
        """predict probabilities for input samples"""
        return self.gb_trees.predict_proba(X)

    def predict(self, X):
        """predict binary labels for input samples"""
        probabilities = self.predict_proba(X)
        return (probabilities > 0.5).int()

    def get_feature_importance(self):
        """get feature importance as a pytorch tensor"""
        return self.feature_importance

    def save(self, path):
        """save the model to disk"""
        model_data = {
            "gb_trees": self.gb_trees,
            "feature_importance": (
                self.feature_importance.detach().cpu().numpy()
                if self.feature_importance is not None
                else None
            ),
        }
        with open(path, "wb") as f:
            pickle.dump(model_data, f)

    def load(self, path):
        """load the model from disk"""
        with open(path, "rb") as f:
            model_data = pickle.load(f)

        self.gb_trees = model_data["gb_trees"]
        if model_data["feature_importance"] is not None:
            self.feature_importance = torch.tensor(model_data["feature_importance"])
