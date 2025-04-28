import streamlit as st

def display_chat_messages():
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def collect_user_input():
    return st.chat_input("Type your message here...")

def display_logs(logs):
    if logs:
        for log in logs:
            st.write(f"**Q:** {log[1]}")
            st.write(f"**A:** {log[2]}")
            st.write(f"**Feedback:** {log[3]}")
            st.write(f"**Timestamp:** {log[4]}")
            st.write("---")
    else:
        st.write("No interactions found.")
