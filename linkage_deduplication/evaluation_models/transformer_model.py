'''
Author: Michael Kurilko
Date: 6/3/2025 (Torch version 6/5/2025)
Description: This module contains the TransformerModel class, which provides
transformer-based similarity scorer using TORCH!
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerModel(nn.Module):
    def __init__(self, embedding_dim=16, vocab_size=256):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.attn_wq = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.attn_wk = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.attn_wv = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.ffn_w1 = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.ffn_w2 = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.gb_weights = nn.Parameter(torch.ones(7))

    def _byte_encode(self, s, max_len=32):
        arr = list(s.encode("utf-8")[:max_len])
        arr = arr + [0] * (max_len - len(arr))
        return torch.tensor(arr, dtype=torch.long)

    def _embed(self, arr):
        return self.embeddings(arr)

    def _self_attention(self, x):
        Q = self.attn_wq(x)
        K = self.attn_wk(x)
        V = self.attn_wv(x)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.embedding_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        return torch.matmul(attn_weights, V)

    def _ffn(self, x):
        return self.ffn_w2(F.relu(self.ffn_w1(x)))

    def _transformer_encode(self, arr):
        x = self._embed(arr)
        x = self._self_attention(x)
        x = self._ffn(x)
        return x.mean(dim=0)

    def transformer_similarity(self, subject1, subject2):
        features = ['first_name', 'middle_name', 'last_name', 'dob', 'dod', 'email', 'phone_number', 'birth_city']
        s1 = " ".join([str(getattr(subject1, f, "")) for f in features])
        s2 = " ".join([str(getattr(subject2, f, "")) for f in features])
        arr1 = self._byte_encode(s1).to(self.embeddings.weight.device)
        arr2 = self._byte_encode(s2).to(self.embeddings.weight.device)
        vec1 = self._transformer_encode(arr1)
        vec2 = self._transformer_encode(arr2)
        dot = (vec1 * vec2).sum()
        norm1 = vec1.norm()
        norm2 = vec2.norm()
        similarity = dot / (norm1 * norm2 + 1e-8)
        return float(((similarity + 1) / 2).item())

    def transformer_similarity_tensor(self, subject1, subject2):
        features = ['first_name', 'middle_name', 'last_name', 'dob', 'dod', 'email', 'phone_number', 'birth_city']
        s1 = " ".join([str(getattr(subject1, f, "")) for f in features])
        s2 = " ".join([str(getattr(subject2, f, "")) for f in features])
        arr1 = self._byte_encode(s1).to(self.embeddings.weight.device)
        arr2 = self._byte_encode(s2).to(self.embeddings.weight.device)
        vec1 = self._transformer_encode(arr1)
        vec2 = self._transformer_encode(arr2)
        dot = (vec1 * vec2).sum()
        norm1 = vec1.norm()
        norm2 = vec2.norm()
        similarity = dot / (norm1 * norm2 + 1e-8)
        return (similarity + 1) / 2  # Tensor in [0,1]

    def train_transformer(self, subject_pairs, labels, epochs=10, lr=1e-3, device="cuda"):
        self.to(device)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        eps = 1e-7
        for epoch in range(epochs):
            total_loss = 0.0
            for (subj1, subj2), label in zip(subject_pairs, labels):
                pred = self.transformer_similarity_tensor(subj1, subj2)
                pred = torch.clamp(pred, eps, 1 - eps)
                target = torch.full(pred.shape, float(label), device=pred.device)
                loss = -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred)).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += float(loss.item())
                if torch.isnan(loss):
                    print(f"NaN detected in loss! pred={pred.cpu().detach().numpy()}, target={target.cpu().detach().numpy()}")
                    break
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(subject_pairs):.4f}")

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, map_location=None):
        self.load_state_dict(torch.load(path, map_location=map_location or "cpu"))

# Example usage:
# model = TransformerModel()
# gb_score = model.gradient_boosted_score(subject1, subject2)
# transformer_score = model.transformer_similarity(subject1, subject2)

'''
# Train your model
model = TransformerModel()
model.train_transformer(subject_pairs, labels, epochs=20, lr=1e-3, device="cuda" if torch.cuda.is_available() else "cpu")

# Save the trained weights
model.save("transformer_weights.npz")


# Later, or in a new script/session
model = TransformerModel()
model.load("transformer_weights.npz")

# Now you can use the trained model for inference
score = model.transformer_similarity(subject1, subject2)
print("Similarity score:", score)
'''
