'''
Author: Michael Kurilko
Date: 6/3/2025
Description: This module contains the ComparisonModel class, which provides
gradient-boosted and transformer-based similarity scorers using tinygrad.
'''

import numpy as np
from tinygrad import Tensor
from tinygrad.nn.optim import SGD

class ComparisonModel:
    def __init__(self):
        # 7 features: first_name, middle_name, last_name, dob, dod, email, birth_city
        self.gb_weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        # Transformer-like parameters
        self.embedding_dim = 16
        self.vocab_size = 256  # Byte-level encoding

        # Embedding matrix (vocab_size x embedding_dim)
        self.embeddings = Tensor.uniform(self.vocab_size, self.embedding_dim)

        # Simple transformer weights (single self-attention head for demonstration)
        self.attn_wq = Tensor.uniform(self.embedding_dim, self.embedding_dim)
        self.attn_wk = Tensor.uniform(self.embedding_dim, self.embedding_dim)
        self.attn_wv = Tensor.uniform(self.embedding_dim, self.embedding_dim)
        self.ffn_w1 = Tensor.uniform(self.embedding_dim, self.embedding_dim)
        self.ffn_w2 = Tensor.uniform(self.embedding_dim, self.embedding_dim)

    def gradient_boosted_score(self, subject1, subject2):
        features = ['first_name', 'middle_name', 'last_name', 'dob', 'dod', 'email', 'birth_city']
        x = []
        for i, feature in enumerate(features):
            attr1 = getattr(subject1, feature, "")
            attr2 = getattr(subject2, feature, "")
            x.append(-1.0 if attr1 != attr2 else 1.0)
        x = np.array(x)
        score = np.dot(self.gb_weights, x)
        return 1 / (1 + np.exp(-score))

    def _byte_encode(self, s, max_len=32):
        # Encode string to byte indices, pad/truncate to max_len
        arr = np.frombuffer(s.encode("utf-8")[:max_len], dtype=np.uint8)
        if len(arr) < max_len:
            arr = np.pad(arr, (0, max_len - len(arr)), constant_values=0)
        return arr

    def _embed(self, arr):
        # arr: numpy array of byte indices
        idx = Tensor(arr.astype(np.int32))
        return self.embeddings[idx].reshape((-1, self.embedding_dim))

    def _self_attention(self, x):
        # x: (seq_len, embedding_dim)
        Q = x @ self.attn_wq
        K = x @ self.attn_wk
        V = x @ self.attn_wv
        attn_scores = (Q @ K.transpose()) / np.sqrt(self.embedding_dim)
        attn_weights = attn_scores.softmax(axis=-1)
        return attn_weights @ V

    def _ffn(self, x):
        return (x @ self.ffn_w1).relu() @ self.ffn_w2

    def _transformer_encode(self, arr):
        # arr: numpy array of byte indices
        x = self._embed(arr)
        x = self._self_attention(x)
        x = self._ffn(x)
        # Pooling: mean over sequence
        return x.mean(axis=0)

    def transformer_similarity(self, subject1, subject2):
        features = ['first_name', 'middle_name', 'last_name', 'dob', 'dod', 'email', 'birth_city']
        s1 = " ".join([str(getattr(subject1, f, "")) for f in features])
        s2 = " ".join([str(getattr(subject2, f, "")) for f in features])
        arr1 = self._byte_encode(s1)
        arr2 = self._byte_encode(s2)
        vec1 = self._transformer_encode(arr1)
        vec2 = self._transformer_encode(arr2)
        # Cosine similarity
        dot = (vec1 * vec2).sum()
        norm1 = (vec1 * vec1).sum().sqrt()
        norm2 = (vec2 * vec2).sum().sqrt()
        similarity = dot / (norm1 * norm2 + 1e-8)
        # Map cosine similarity [-1,1] to [0,1] and convert to float
        return float(((similarity + 1) / 2).item())
    
    def transformer_similarity_tensor(self, subject1, subject2):
        # Returns a Tensor, not float, for training
        features = ['first_name', 'middle_name', 'last_name', 'dob', 'dod', 'email', 'birth_city']
        s1 = " ".join([str(getattr(subject1, f, "")) for f in features])
        s2 = " ".join([str(getattr(subject2, f, "")) for f in features])
        arr1 = self._byte_encode(s1)
        arr2 = self._byte_encode(s2)
        vec1 = self._transformer_encode(arr1)
        vec2 = self._transformer_encode(arr2)
        dot = (vec1 * vec2).sum()
        norm1 = (vec1 * vec1).sum().sqrt()
        norm2 = (vec2 * vec2).sum().sqrt()
        similarity = dot / (norm1 * norm2 + 1e-8)
        return (similarity + 1) / 2  # Tensor in [0,1]

    def train_transformer(self, subject_pairs, labels, epochs=10, lr=1e-3):
        # subject_pairs: list of (subject1, subject2)
        # labels: list of 0/1
        params = [
            self.embeddings, self.attn_wq, self.attn_wk, self.attn_wv, self.ffn_w1, self.ffn_w2
        ]
        opt = SGD(params, lr=lr)
        for epoch in range(epochs):
            total_loss = 0.0
            for (subj1, subj2), label in zip(subject_pairs, labels):
                pred = self.transformer_similarity_tensor(subj1, subj2)
                target = Tensor([label], requires_grad=False)
                # Binary cross-entropy loss
                loss = -(target * pred.log() + (1 - target) * (1 - pred).log()).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += float(loss.item())
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(subject_pairs):.4f}")
    
    def save(self, path):
        np.savez(path,
            embeddings=self.embeddings.numpy(),
            attn_wq=self.attn_wq.numpy(),
            attn_wk=self.attn_wk.numpy(),
            attn_wv=self.attn_wv.numpy(),
            ffn_w1=self.ffn_w1.numpy(),
            ffn_w2=self.ffn_w2.numpy()
        )

    def load(self, path):
        data = np.load(path)
        self.embeddings = Tensor(data['embeddings'])
        self.attn_wq = Tensor(data['attn_wq'])
        self.attn_wk = Tensor(data['attn_wk'])
        self.attn_wv = Tensor(data['attn_wv'])
        self.ffn_w1 = Tensor(data['ffn_w1'])
        self.ffn_w2 = Tensor(data['ffn_w2'])

# Example usage:
# model = ComparisonModel()
# gb_score = model.gradient_boosted_score(subject1, subject2)
# transformer_score = model.transformer_similarity(subject1, subject2)

"""
# Train your model
model = ComparisonModel()
model.train_transformer(subject_pairs, labels, epochs=20, lr=1e-3)

# Save the trained weights
model.save("transformer_weights.npz")


# Later, or in a new script/session
model = ComparisonModel()
model.load("transformer_weights.npz")

# Now you can use the trained model for inference
score = model.transformer_similarity(subject1, subject2)
print("Similarity score:", score)
"""