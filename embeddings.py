from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma
from config import HF_API_TOKEN, EMBEDDING_MODEL, CHROMA_DB_DIR, CHUNK_SIZE, CHUNK_OVERLAP


def get_embedding_model():
    """Get the HuggingFace Inference API embedding model."""
    return HuggingFaceEndpointEmbeddings(
        model=EMBEDDING_MODEL,
        huggingfacehub_api_token=HF_API_TOKEN,
    )


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_text(text)


def video_exists_in_db(video_id: str) -> bool:
    """Check if a video has already been processed and stored."""
    embedding_model = get_embedding_model()
    vectorstore = Chroma(
        collection_name=video_id,
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embedding_model,
    )
    return vectorstore._collection.count() > 0


def store_transcript(video_id: str, transcript_text: str):
    """Chunk the transcript and store embeddings in ChromaDB."""
    chunks = chunk_text(transcript_text)
    embedding_model = get_embedding_model()

    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embedding_model,
        collection_name=video_id,
        persist_directory=CHROMA_DB_DIR,
        metadatas=[{"video_id": video_id, "chunk_index": i} for i in range(len(chunks))],
    )
    return vectorstore


def get_vectorstore(video_id: str):
    """Load an existing vectorstore for a video."""
    embedding_model = get_embedding_model()
    return Chroma(
        collection_name=video_id,
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embedding_model,
    )
