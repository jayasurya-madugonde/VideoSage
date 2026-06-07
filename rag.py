from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from config import GROQ_API_KEY, LLM_MODEL, LLM_TEMPERATURE, RETRIEVAL_TOP_K
from embeddings import get_vectorstore

SYSTEM_PROMPT = """You are VideoSage, a helpful assistant that answers questions about YouTube videos based on their transcripts.
Use ONLY the following transcript excerpts to answer the question. If the answer is not found in the provided context, say "I couldn't find the answer in this video's transcript."

Context from the video transcript:
{context}
"""


def get_llm():
    """Get the Groq LLM."""
    return ChatGroq(
        model=LLM_MODEL,
        api_key=GROQ_API_KEY,
        temperature=LLM_TEMPERATURE,
    )


def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


def get_rag_chain(video_id: str):
    """Build and return the RAG chain for a given video."""
    vectorstore = get_vectorstore(video_id)
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_TOP_K})

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    llm = get_llm()

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


def ask_question(video_id: str, question: str) -> str:
    """Ask a question about a video and get an answer."""
    chain = get_rag_chain(video_id)
    return chain.invoke(question)
