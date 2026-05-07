import streamlit as st
from api import delete_document

def show():
    st.title("🗑️ Manage Documents")
    st.markdown("Delete documents from the RAG knowledge base.")

    st.subheader("Delete a Document")
    st.caption("Enter the exact filename including extension (e.g. chromadb.txt)")

    doc_id = st.text_input(
        "Document filename",
        placeholder="e.g. chromadb.txt"
    )

    if st.button("Delete Document", type="primary"):
        if not doc_id.strip():
            st.error("Please enter a filename!")
        else:
            with st.spinner(f"Deleting {doc_id}..."):
                result = delete_document(doc_id)

            if "error" in result:
                st.error(f"Delete failed: {result['error']}")
            elif result.get("detail"):
                st.warning(f"⚠️ {result['detail']}")
            else:
                st.success(f"✅ {result.get('message', 'Deleted successfully')}")
                st.metric("Chunks Deleted", result.get("chunks_deleted", 0))