import streamlit as st
from api import delete_document, list_documents

def show():
    st.title("🗑️ Manage Documents")
    st.markdown("Review and remove documents in the RAG knowledge base.")

    notice = st.session_state.pop("document_notice", None)
    if notice:
        st.success(notice)

    if st.button("Refresh documents"):
        st.rerun()

    result = list_documents()
    if "error" in result:
        st.error(f"Could not load documents: {result['error']}")
        return
    if "detail" in result:
        st.error(f"Could not load documents: {result['detail']}")
        return

    documents = result.get("documents", [])
    if not documents:
        st.info("No indexed documents found.")
        return

    st.caption(f"{len(documents)} indexed document(s)")
    for document in documents:
        doc_id = document["doc_id"]
        size = document.get("size_bytes")
        size_text = f" · {size:,} bytes" if size is not None else " · file missing"
        with st.container(border=True):
            st.subheader(doc_id)
            st.caption(f"{document['chunks_indexed']} chunks{size_text}")
            confirmed = st.checkbox(
                f"Confirm deletion of {doc_id}", key=f"confirm_{doc_id}"
            )
            if st.button("Delete", key=f"delete_{doc_id}", disabled=not confirmed):
                with st.spinner(f"Deleting {doc_id}..."):
                    deletion = delete_document(doc_id)
                if "error" in deletion or "detail" in deletion:
                    st.error(deletion.get("error") or deletion["detail"])
                else:
                    st.session_state["document_notice"] = deletion.get(
                        "message", f"Deleted {doc_id}"
                    )
                    st.rerun()
