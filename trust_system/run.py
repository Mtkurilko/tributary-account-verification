'''
Author: Michael Kurilko
Date: 7/21/2025
Description: The file that runs the representation of the trust system.
'''

import streamlit as st
from system import TrustSystem
import os

st.set_page_config(layout="wide")
st.title("User Trust Interaction Dashboard")

# === Initialize or refresh TrustSystem ===
if "trust_system" not in st.session_state:
    st.session_state.trust_system = TrustSystem()

trust_system = st.session_state.trust_system

# === Sidebar: User Management ===
st.sidebar.header("Manage Users")
new_user = st.sidebar.text_input("Add New User")
if st.sidebar.button("Add User"):
    if new_user.strip():
        trust_system.add_user(new_user.strip())
        st.sidebar.success(f"User '{new_user.strip()}' added.")

usernames = list(trust_system.users.keys())

# === Sidebar: Import / Export State ===
st.sidebar.header("Data I/O")
if st.sidebar.button("Export to JSON"):
    trust_system.export_to_json()
    st.sidebar.success("Exported to trust_state.json")

if st.sidebar.button("Import from JSON"):
    if os.path.exists("trust_system/trust_state.json"):
        trust_system.import_from_json()
        st.session_state.trust_system = trust_system  # Refresh
        st.rerun()
    else:
        st.sidebar.error("trust_state.json not found.")

# === Main Panel: Interaction Toolkit ===
if len(usernames) >= 2:
    st.subheader("Simulate User Interactions")

    col1, col2 = st.columns(2)
    actor = col1.selectbox("Acting User", usernames)
    target = col2.selectbox("Target User", [u for u in usernames if u != actor])

    col3, col4, col5 = st.columns(3)
    if col3.button("Vet"):
        trust_system.vet_user(by=actor, target=target)
        st.success(f"{actor} vetted {target}")

    if col4.button("Accept"):
        trust_system.accept_user(by=actor, target=target)
        st.success(f"{actor} accepted {target}")

    if col5.button("Report"):
        trust_system.report_user(by=actor, target=target)
        st.warning(f"{actor} reported {target}")

else:
    st.info("Add at least two users to begin simulating interactions.")

# === Trust Scores & History Table ===
st.subheader("User Trust Overview")

for username in usernames:
    user = trust_system.get_user(username)
    trust = user.get_recent_trust()
    with st.expander(f"{username} — Trust Score: {trust}"):
        st.text("Interaction Log:")
        if user.interactions:
            for i in user.interactions[::-1]:
                st.markdown(
                    f"- `{i.timestamp}`: **{i.actor}** {i.action} **{i.target}** ({'+' if i.weight >= 0 else ''}{i.weight})"
                )
        else:
            st.write("No interactions yet.")

# === Spam Detection ===
st.subheader("Spam Report Detection")
spammers = trust_system.detect_spam_reporters()
if spammers:
    st.error(f"Users flagged as spamming reports: {', '.join(spammers)}")
else:
    st.success("No spam reporters detected.")
