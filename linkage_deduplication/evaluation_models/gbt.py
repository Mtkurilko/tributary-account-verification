"""
gradient boosted tree implementation for pytorch
"""

import torch
import torch.nn as nn
import numpy as np
import pickle

PROB_BARRIER = 0.5  # threshold for binary classification
N_ESTIMATORS = 500  # default number of trees in the ensemble
LEARNING_RATE = 0.1  # default learning rate for boosting
MAX_DEPTH = 5  # default maximum depth of each tree

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
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        max_depth=MAX_DEPTH,
        min_samples_split=2,
        min_samples_leaf=1,
        device=None,
        verbose=True,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.trees = []
        self.initial_prediction = 0.0
        self.device = self._setup_device(device)

    def _setup_device(self, device):
        """setup and validate device selection"""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if isinstance(device, str):
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            device = torch.device(device)

        if device.type == "cuda" and not torch.cuda.is_available():
            device = torch.device("cpu")

        return device

    def to(self, device):
        """move model to specified device"""
        self.device = self._setup_device(device)
        return self

    def cuda(self):
        """move model to cuda device"""
        return self.to("cuda")

    def cpu(self):
        """move model to cpu device"""
        return self.to("cpu")

    def _move_to_device(self, tensor):
        """move tensor to the correct device"""
        if isinstance(tensor, torch.Tensor):
            return tensor.to(self.device)
        elif isinstance(tensor, np.ndarray):
            return torch.tensor(tensor, dtype=torch.float32, device=self.device)
        else:
            return torch.tensor(tensor, dtype=torch.float32, device=self.device)

    def fit(self, X, y):
        """fit data to tree"""
        X = self._move_to_device(X)
        y = self._move_to_device(y)

        self.initial_prediction = torch.mean(y).item()
        predictions = torch.full((len(y),), self.initial_prediction, device=self.device)

        for i in range(self.n_estimators):
            # calculate residuals
            residuals = y - predictions

            # fit tree to residuals
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
            )
            tree.fit(X, residuals)
            self.trees.append(tree)

            # update predictions
            tree_preds = torch.tensor(
                tree.predict(X), device=self.device, dtype=torch.float32
            )
            predictions += self.learning_rate * tree_preds

    def predict(self, X):
        """predict values for input samples"""
        X = self._move_to_device(X)

        predictions = torch.full((len(X),), self.initial_prediction, device=self.device)

        for tree in self.trees:
            tree_predictions = torch.tensor(
                tree.predict(X), device=self.device, dtype=torch.float32
            )
            predictions += self.learning_rate * tree_predictions

        return predictions

    def predict_proba(self, X):
        """predict probabilities using sigmoid activation"""
        raw_predictions = self.predict(X)
        temperature = 4.0  # temperature for scaling logits
        return torch.sigmoid(temperature * raw_predictions)


class GradientBoostedClassifier:
    """gradient boosted trees for binary classification"""

    def __init__(
        self,
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        max_depth=MAX_DEPTH,
        min_samples_split=2,
        min_samples_leaf=1,
        device=None,
        verbose=True,
    ):
        self.gb_trees = GradientBoostedTrees(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            device=device,
            verbose=verbose,
        )
        self.device = self.gb_trees.device
        self.learning_rate = learning_rate

    def to(self, device):
        """move model to specified device"""
        self.gb_trees.to(device)
        self.device = self.gb_trees.device
        return self

    def cuda(self):
        """move model to cuda device"""
        return self.to("cuda")

    def cpu(self):
        """Move model to cpu device"""
        return self.to("cpu")

    def fit(self, X, y):
        """fit data to tree with classifier"""
        # convert binary labels to logits for training
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32, device=self.device)
        else:
            y = y.to(self.device)

        y_logits = torch.where(
            y == 1,
            torch.tensor(1.0, device=self.device),
            torch.tensor(-1.0, device=self.device),
        )

        return self.gb_trees.fit(X, y_logits)

    def train_gbt(self, X, y, epochs=1):
        """
        Train the classifier over multiple epochs, adding trees each epoch.
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        else:
            X = X.to(self.device)

        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32, device=self.device)
        else:
            y = y.to(self.device)

        y_logits = torch.where(
            y == 1,
            torch.tensor(1.0, device=self.device),
            torch.tensor(-1.0, device=self.device),
        )

        # Save original number of trees per epoch
        trees_per_epoch = max(1, self.gb_trees.n_estimators // epochs)
        total_trees = 0
        self.gb_trees.trees = []
        self.gb_trees.initial_prediction = torch.mean(y_logits).item()
        predictions = torch.full((len(y_logits),), self.gb_trees.initial_prediction, device=self.device)

        for epoch in range(epochs):
            for i in range(trees_per_epoch):
                # calculate residuals
                residuals = y_logits - predictions

                # fit tree to residuals
                tree = DecisionTree(
                    max_depth=self.gb_trees.max_depth,
                    min_samples_split=self.gb_trees.min_samples_split,
                    min_samples_leaf=self.gb_trees.min_samples_leaf,
                )
                tree.fit(X, residuals)
                self.gb_trees.trees.append(tree)

                # update predictions
                tree_preds = torch.tensor(
                    tree.predict(X), device=self.device, dtype=torch.float32
                )
                predictions += self.gb_trees.learning_rate * tree_preds
                total_trees += 1

            # Calculate loss and accuracy after this epoch
            loss = torch.nn.functional.mse_loss(predictions, y_logits).item()
            temperature = 4.0  # temperature for scaling logits
            probs = torch.sigmoid(temperature*predictions)
            preds = (probs > PROB_BARRIER).float()
            accuracy = (preds == y).float().mean().item()
            print(f"Epoch {epoch+1}/{epochs}, Trees: {total_trees}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

        # Update n_estimators to reflect total trees
        self.gb_trees.n_estimators = total_trees

    def predict_proba(self, X):
        """predict probabilities for input samples"""
        return self.gb_trees.predict_proba(X)

    def predict(self, X):
        """predict binary labels for input samples"""
        probabilities = self.predict_proba(X)
        return (probabilities > PROB_BARRIER).int()

    def save(self, path):
        """save the model to disk"""
        model_data = {
            "gb_trees": self.gb_trees,
        }
        with open(path, "wb") as f:
            pickle.dump(model_data, f)

    def load(self, path):
        """load the model from disk"""
        with open(path, "rb") as f:
            model_data = pickle.load(f)
        self.gb_trees = model_data["gb_trees"]
