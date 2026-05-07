import streamlit as st
from api import train

def show():
    st.title("🎓 Train Vanna")
    st.markdown("Add new Q&A pairs to improve SQL generation.")

    with st.form("train_form"):
        question = st.text_input(
            "Question",
            placeholder="e.g. How many students enrolled last month?"
        )
        sql = st.text_area(
            "SQL",
            placeholder="e.g. SELECT COUNT(*) FROM enrollments WHERE purchase_date >= date_trunc('month', current_date - interval '1 month')",
            height=150
        )
        submitted = st.form_submit_button("Train", type="primary")

    if submitted:
        if not question.strip():
            st.error("Question cannot be empty!")
        elif not sql.strip():
            st.error("SQL cannot be empty!")
        else:
            with st.spinner("Training..."):
                result = train(question, sql)

            if "error" in result:
                st.error(f"Training failed: {result['error']}")
            else:
                st.success("✅ Training successful!")
                st.json({
                    "question": result.get("question"),
                    "sql": result.get("sql")
                })