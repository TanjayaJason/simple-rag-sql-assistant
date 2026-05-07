import streamlit as st
from api import upload_document, reindex

def show():
    st.title("📄 Upload Document")
    st.markdown("Upload a document to the RAG knowledge base.")

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["txt", "pdf", "docx"],
        help="Supported formats: TXT, PDF, DOCX"
    )

    if uploaded_file:
        st.info(f"File selected: **{uploaded_file.name}** ({uploaded_file.size} bytes)")

        if st.button("Upload & Index", type="primary"):
            with st.spinner("Uploading and indexing..."):
                result = upload_document(
                    uploaded_file.read(),
                    uploaded_file.name
                )

            if "error" in result:
                st.error(f"Upload failed: {result['error']}")
            else:
                st.success(f"✅ {result.get('message', 'Upload successful')}")
                st.metric("Chunks Indexed", result.get("chunks_indexed", 0))

    st.markdown("---")
    st.subheader("🔄 Reindex All Documents")
    st.caption("Use this if you manually added or modified files in the docs folder.")

    if st.button("Reindex All", type="secondary"):
        with st.spinner("Reindexing all documents..."):
            result = reindex()
        if "error" in result:
            st.error(f"Reindex failed: {result['error']}")
        else:
            st.success(f"✅ {result.get('message', 'Reindex complete')}")
            st.metric("Total Chunks Indexed", result.get("chunks_indexed", 0))