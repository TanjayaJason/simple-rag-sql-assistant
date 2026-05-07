# frontend/main.py
import streamlit as st
from api import get_health

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Agentic RAG + Text2SQL",
    page_icon="🤖",
    layout="wide"
)

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("RAG + Text2SQL")
st.sidebar.markdown("---")

# Health check
health = get_health()
if "error" in health:
    st.sidebar.error("❌ Backend offline")
else:
    st.sidebar.success("✅ Backend online")

st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["💬 Ask", "📄 Upload", "📋 History", "🎓 Train", "🗑️ Documents"]
)

# -----------------------------
# ROUTING
# -----------------------------
if page == "💬 Ask":
    from views.ask import show
    show()
elif page == "📄 Upload":
    from views.upload import show
    show()
elif page == "📋 History":
    from views.history import show
    show()
elif page == "🎓 Train":
    from views.train import show
    show()
elif page == "🗑️ Documents":
    from views.documents import show
    show()