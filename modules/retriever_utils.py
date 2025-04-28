from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_groq import ChatGroq

# Function to initialize the Groq LLM
def initialize_llm(api_key):
    return ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")

# Create the history-aware retriever
def create_retriever_chain(llm, retriever):
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question, "
        "formulate a standalone question that can be understood "
        "without the chat history."
    )
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    
    return create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

# Create the RAG pipeline
def create_rag_pipeline(llm, retriever):
    history_aware_retriever = create_retriever_chain(llm, retriever)

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the retrieved context to answer concisely. "
        "If you don't know the answer, say that you don't know."
        "\n\n{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain
