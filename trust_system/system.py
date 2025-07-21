'''
Author: Michael Kurilko
Date: 7/21/2025
Description: This script contains the trust quantifiers and their evaluation.
'''

from dataclasses import dataclass, field
from typing import Dict, List
from datetime import datetime, timedelta
import json
import math

# ======= CONFIGURABLE CONSTANTS ======= #

INITIAL_TRUST = 50.0
TRUST_DECAY_DAYS = 30             # Days for full decay effect
MAX_TRUST = 100.0
MIN_TRUST = 0.0

TRUST_GAIN_ON_VET = 8             # Base trust gain when vetted by another user
TRUST_LOSS_ON_REPORT = 15         # Base trust loss when reported
TRUST_GAIN_ON_ACCEPT = 4          # Base gain for social acceptance
SPAM_REPORT_THRESHOLD = 5         # Reports in short time frame to flag spam
TRUST_WEIGHT_MULTIPLIER = 0.01    # Influence multiplier based on vetting user's trust

# ======= USER CLASS ======= #

@dataclass
class Interaction:
    actor: str
    target: str
    action: str  # vet, report, accept
    timestamp: str
    weight: float

@dataclass
class User:
    username: str
    trust_score: float = INITIAL_TRUST
    interactions: List[Interaction] = field(default_factory=list)

    def get_recent_trust(self, current_time=None):
        if current_time is None:
            current_time = datetime.utcnow()
        total = self.trust_score
        for interaction in self.interactions:
            t = datetime.fromisoformat(interaction.timestamp)
            days_ago = (current_time - t).days
            decay_factor = math.exp(-days_ago / TRUST_DECAY_DAYS)
            total += interaction.weight * decay_factor
        return min(MAX_TRUST, max(MIN_TRUST, round(total, 2)))

    def log_interaction(self, action: str, target: str, weight: float):
        self.interactions.append(
            Interaction(
                actor=self.username,
                target=target,
                action=action,
                timestamp=datetime.utcnow().isoformat(),
                weight=weight
            )
        )

# ======= TRUST SYSTEM ======= #

class TrustSystem:
    def __init__(self):
        self.users: Dict[str, User] = {}

    def add_user(self, username: str):
        if username not in self.users:
            self.users[username] = User(username=username)

    def get_user(self, username: str) -> User:
        if username not in self.users:
            raise ValueError(f"User '{username}' does not exist.")
        return self.users[username]

    def vet_user(self, by: str, target: str):
        self._validate_users(by, target)
        by_user = self.get_user(by)
        target_user = self.get_user(target)

        weight = TRUST_GAIN_ON_VET * (1 + by_user.get_recent_trust() * TRUST_WEIGHT_MULTIPLIER)
        target_user.log_interaction("vetted_by", by, weight)
        by_user.log_interaction("vetted", target, 0)

    def report_user(self, by: str, target: str):
        self._validate_users(by, target)
        by_user = self.get_user(by)
        target_user = self.get_user(target)

        target_user.log_interaction("reported_by", by, -TRUST_LOSS_ON_REPORT)
        by_user.log_interaction("reported", target, 0)

    def accept_user(self, by: str, target: str):
        self._validate_users(by, target)
        target_user = self.get_user(target)
        by_user = self.get_user(by)

        weight = TRUST_GAIN_ON_ACCEPT * (1 + by_user.get_recent_trust() * TRUST_WEIGHT_MULTIPLIER)
        target_user.log_interaction("accepted_by", by, weight)
        by_user.log_interaction("accepted", target, 0)

    def _validate_users(self, *usernames):
        for name in usernames:
            if name not in self.users:
                raise ValueError(f"User '{name}' does not exist.")

    def detect_spam_reporters(self) -> List[str]:
        spam_users = []
        now = datetime.utcnow()
        for user in self.users.values():
            recent_reports = [i for i in user.interactions if i.action == "reported" and
                              (now - datetime.fromisoformat(i.timestamp)) < timedelta(hours=1)]
            if len(recent_reports) >= SPAM_REPORT_THRESHOLD:
                spam_users.append(user.username)
        return spam_users

    def export_to_json(self, path="trust_system/trust_state.json"):
        data = {
            "users": {
                username: {
                    "trust_score": user.trust_score,
                    "interactions": [
                        {
                            "actor": i.actor,
                            "target": i.target,
                            "action": i.action,
                            "timestamp": i.timestamp,
                            "weight": i.weight
                        }
                        for i in user.interactions
                    ]
                } for username, user in self.users.items()
            }
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def import_from_json(self, path="trust_system/trust_state.json"):
        with open(path, "r") as f:
            data = json.load(f)
        self.users.clear()
        for username, info in data["users"].items():
            user = User(username=username, trust_score=info["trust_score"])
            for i in info["interactions"]:
                user.interactions.append(Interaction(**i))
            self.users[username] = user

    def reset_user(self, username: str):
        if username in self.users:
            self.users[username].reset()

    def reset_all_users(self):
        for user in self.users.values():
            user.reset()
    
    def delete_user(self, username: str):
        if username in self.users:
            del self.users[username]

    def delete_all_users(self):
        self.users.clear()