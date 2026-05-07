import streamlit as st
from api import ask_question

def show():
    st.title("💬 Ask a Question")
    st.markdown("Ask anything — the system will automatically route to SQL or RAG.")

    # New chat button
    if st.button("🆕 New Chat", type="secondary"):
        st.session_state.messages = []
        st.rerun()

    # Chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if question := st.chat_input("Ask a question..."):
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = ask_question(question)

            if "error" in result:
                st.error(f"Error: {result['error']}")
            else:
                st.markdown(result.get("answer", "No answer"))

                col1, col2, col3 = st.columns(3)
                with col1:
                    tool = result.get("tool_used", "unknown")
                    color = "🟦" if tool == "SQL" else "🟩"
                    st.caption(f"{color} Tool: **{tool}**")
                with col2:
                    st.caption(f"⏱ Time: **{result.get('response_time_seconds', 0)}s**")
                with col3:
                    confidence = result.get("confidence_note") or result.get("confidence", "")
                    st.caption(f"🎯 Confidence: **{confidence}**")

                if result.get("sql"):
                    with st.expander("🔍 Generated SQL"):
                        st.code(result["sql"], language="sql")

                if result.get("sources"):
                    with st.expander("📚 Sources"):
                        for source in result["sources"]:
                            st.markdown(f"- `{source}`")

                if result.get("result"):
                    with st.expander("📊 Raw Data"):
                        st.dataframe(result["result"])

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result.get("answer", "No answer")
                })