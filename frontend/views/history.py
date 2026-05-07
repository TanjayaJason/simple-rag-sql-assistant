import streamlit as st
from api import get_history

def show():
    st.title("📋 Conversation History")

    limit = st.slider("Number of records", min_value=1, max_value=10, value=1)

    if st.button("Load History", type="primary"):
        with st.spinner("Loading..."):
            result = get_history(limit=limit)

        if "error" in result:
            st.error(f"Error: {result['error']}")
        elif result.get("count", 0) == 0:
            st.info("No history found.")
        else:
            st.metric("Total Records", result["count"])
            st.markdown("---")

            for record in result["history"]:
                with st.expander(f"❓ {record['question'][:80]}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        tool = record.get("tool_used", "")
                        color = "🟦" if tool == "SQL" else "🟩"
                        st.caption(f"{color} Tool: **{tool}**")
                    with col2:
                        st.caption(f"🕐 {record.get('created_at', '')}")

                    st.markdown("**Question:**")
                    st.markdown(record["question"])

                    st.markdown("**Answer:**")
                    st.markdown(record["answer"])

                    if record.get("sql_generated"):
                        st.markdown("**SQL:**")
                        st.code(record["sql_generated"], language="sql")