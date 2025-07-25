'''
Author: Michael Kurilko
Date: 7/21/2025
Description: This script contains the trust quantifiers and their evaluation.
'''

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import json
import math

# ======= CONFIGURABLE CONSTANTS ======= #

INITIAL_TRUST = 50.0
TRUST_DECAY_DAYS = 30             # Days for full decay effect (not in use yet)
MAX_TRUST = 100.0
MIN_TRUST = 0.0
CURVE_EXPONENT = 2.0  # steeper makes perfection harder

TRUST_GAIN_ON_VET = 8             # Base trust gain when vetted by another user
TRUST_LOSS_ON_VETDENY = 5         # Base trust loss when vetting is denied
TRUST_LOSS_ON_REPORT = 15         # Base trust loss when reported
TRUST_GAIN_ON_ACCEPT = 4          # Base gain for social acceptance
SPAM_REPORT_THRESHOLD = 5         # Reports in short time frame to flag spam
SPAM_ACCEPTANCE_THRESHOLD = 30     # Acceptances in short time frame to flag spam
SPAM_PENALTY = 20.0               # Penalty for spamming reports/acceptances
TRUST_WEIGHT_MULTIPLIER = 0.05    # Influence multiplier based on vetting user's trust

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

    def apply_trust_gain(self, base_delta: float):
        scaling_factor = min(1, max(0, 1 - ((self.trust_score-INITIAL_TRUST) / (MAX_TRUST-INITIAL_TRUST))))
        gain = base_delta * (scaling_factor ** CURVE_EXPONENT)
        self.trust_score = min(MAX_TRUST, self.trust_score + gain)
        return gain

    def apply_trust_penalty(self, amount: float, by_user: Optional["User"] = None):
        if by_user is not None:
            by_trust = max(by_user.trust_score, 1e-3)  # Prevent division by zero
            target_trust = max(self.trust_score, 1e-3)

            trust_ratio = by_trust / (target_trust + 1)
            scaling_factor = trust_ratio ** (1 / CURVE_EXPONENT)
            amount *= scaling_factor

        self.trust_score = max(MIN_TRUST, self.trust_score - amount)
        return amount

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
    
    def reset(self):
        self.trust_score = INITIAL_TRUST
        self.interactions.clear()

# ======= TRUST SYSTEM ======= #

class TrustSystem:
    def __init__(self):
        self.users: Dict[str, User] = {}
        self.spam_users: List[str] = []

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

        # Only let users vet a person once
        if any((i.action == "vetted" or i.action == "vet_denied") and i.target == target for i in by_user.interactions):
            return False
        else:
            influence = max(0, TRUST_GAIN_ON_VET * (1 + (by_user.trust_score-INITIAL_TRUST) * TRUST_WEIGHT_MULTIPLIER))
            applied = target_user.apply_trust_gain(influence)
            target_user.log_interaction("vetted_by", by, applied)
            by_user.log_interaction("vetted", target, 0)
            return True
        
    def deny_vet_user(self, by: str, target: str):
        self._validate_users(by, target)
        by_user = self.get_user(by)
        target_user = self.get_user(target)

        # Only let users deny a person once
        if any((i.action == "vet_denied" or i.action == "vetted") and i.target == target for i in by_user.interactions):
            return False
        else:
            penalty = max(0, TRUST_LOSS_ON_VETDENY * (1 + (by_user.trust_score-INITIAL_TRUST) * TRUST_WEIGHT_MULTIPLIER))
            amount = target_user.apply_trust_penalty(penalty, by_user)
            target_user.log_interaction("vet_denied_by", by, -amount)
            by_user.log_interaction("vet_denied", target, 0)
            return True

    def report_user(self, by: str, target: str):
        self._validate_users(by, target)
        by_user = self.get_user(by)
        target_user = self.get_user(target)
        spam_users = self.detect_spam_reporters()

        # Prevent spam users from reporting
        if by_user.username in spam_users:
            return False

        # Check if user already reported this target
        already_reported = any(i.action == "reported" and i.target == target for i in by_user.interactions)
        if already_reported:
            return False

        amount = target_user.apply_trust_penalty(TRUST_LOSS_ON_REPORT, by_user)
        target_user.log_interaction("reported_by", by, -amount)
        by_user.log_interaction("reported", target, 0)
        return True

    def accept_user(self, by: str, target: str):
        self._validate_users(by, target)
        target_user = self.get_user(target)
        by_user = self.get_user(by)

        influence = max(0, TRUST_GAIN_ON_ACCEPT * (1 + (by_user.trust_score-INITIAL_TRUST) * TRUST_WEIGHT_MULTIPLIER))
        applied = target_user.apply_trust_gain(influence)
        target_user.log_interaction("accepted_by", by, applied)
        by_user.log_interaction("accepted", target, 0)

    def _validate_users(self, *usernames):
        for name in usernames:
            if name not in self.users:
                raise ValueError(f"User '{name}' does not exist.")

    def detect_spam_reporters(self) -> List[str]:
        now = datetime.utcnow()

        if not hasattr(self, "spam_users"):
            self.spam_users = []

        for user in self.users.values():
            # Skip users already marked as spammers
            if user.username in self.spam_users:
                continue

            recent_reports = [
                i for i in user.interactions
                if i.action == "reported" and (now - datetime.fromisoformat(i.timestamp)) < timedelta(hours=1)
            ]
            recent_acceptances = [
                i for i in user.interactions
                if i.action == "accepted" and (now - datetime.fromisoformat(i.timestamp)) < timedelta(hours=1)
            ]

            # Only apply penalty once if threshold met and hasn't been penalized already
            if (len(recent_reports) >= SPAM_REPORT_THRESHOLD or
                len(recent_acceptances) >= SPAM_ACCEPTANCE_THRESHOLD):

                already_penalized = any(
                    i.action == "spam_detected" and
                    (now - datetime.fromisoformat(i.timestamp)) < timedelta(hours=1)
                    for i in user.interactions
                )

                if not already_penalized:
                    user.apply_trust_penalty(SPAM_PENALTY)
                    user.log_interaction("spam_detected", "system", -SPAM_PENALTY)
                    self.spam_users.append(user.username)

        return self.spam_users

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