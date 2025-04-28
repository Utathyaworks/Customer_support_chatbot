import streamlit as st
from modules.embedding_utils import load_vectorstore
from modules.retriever_utils import initialize_llm, create_rag_pipeline
from modules.chat_utils import get_session_history
from modules.ui_components import display_chat_messages, collect_user_input, display_logs
from sql_lite_database.database import create_connection, create_table, insert_interaction, get_all_logs
from langchain_core.runnables.history import RunnableWithMessageHistory

st.title("Conversational RAG with PDF Uploads & Chat History")
st.write("Upload PDFs and chat with their content.")

# Get API Key
api_key = st.text_input("Enter your Groq API key:", type="password")

# Initialize database
conn = create_connection()
create_table(conn)

if api_key:
    llm = initialize_llm(api_key)
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever()
    
    rag_chain = create_rag_pipeline(llm, retriever)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain, get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    session_id = "default_session"

    if 'store' not in st.session_state:
        st.session_state.store = {}

    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []

    display_chat_messages()

    user_input = collect_user_input()

    if user_input:
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        response = conversational_rag_chain.invoke({"input": user_input}, config={"configurable": {"session_id": session_id}})
        assistant_response = response['answer']

        st.session_state.chat_messages.append({"role": "assistant", "content": assistant_response})
        insert_interaction(user_input, assistant_response, "No feedback", conn)

        logs = get_all_logs(conn)
        display_logs(logs)

        st.rerun()
else:
    st.warning("Please enter the Groq API key.")
