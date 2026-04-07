"""
Text Chunker and Embedder Tool
Splits markdown texts into chunks using header-aware splitting and generates embeddings
"""

from typing import List
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings


def chunk_and_embed_texts(
    documents: List[dict],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    model_name: str = "all-MiniLM-L6-v2"
) -> List[dict]:
    """
    Split texts into chunks using header-aware splitting and generate embeddings.

    Uses a two-stage approach:
    1. MarkdownHeaderTextSplitter - splits on headers and preserves section metadata
    2. RecursiveCharacterTextSplitter - ensures chunks don't exceed size limits

    Args:
        documents: List of document dictionaries, each containing:
            - text: The markdown text content to chunk and embed
            - file_name: Name of the source PDF file
            - index: Index of the document
        chunk_size: Maximum size of each chunk in characters (default: 1000)
        chunk_overlap: Number of characters to overlap between chunks (default: 200)
        model_name: Name of the sentence transformer model to use (default: "all-MiniLM-L6-v2")

    Returns:
        List of chunk dictionaries, each with:
            - text: The chunk text content
            - embedding: The embedding vector for the chunk
            - file_name: Name of the source PDF file
            - document_index: Index of the original document
            - chunk_index: Index of this chunk within its source document
            - headers: Dict of header metadata (h1, h2, h3, h4) for this chunk's section

    Raises:
        ValueError: If documents list is empty
    """
    if not documents:
        raise ValueError("Documents list cannot be empty")

    # Stage 1: Header-aware splitter (preserves section metadata)
    headers_to_split_on = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
        ("####", "h4"),
    ]
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False
    )

    # Stage 2: Size-enforcing splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n"]
    )

    # Initialize the embedding model
    embedding_function = SentenceTransformerEmbeddings(model_name=model_name)

    chunks = []

    for document in documents:
        text = document.get("text", "")
        file_name = document.get("file_name", "")
        document_index = document.get("index", 0)

        # Skip empty texts
        if not text:
            continue

        # Stage 1: Split by headers (returns Documents with metadata)
        header_splits = header_splitter.split_text(text)

        # Stage 2: Split any oversized sections
        final_splits = text_splitter.split_documents(header_splits)

        for chunk_index, split in enumerate(final_splits):
            # Generate embedding for this chunk
            embedding = embedding_function.embed_query(split.page_content)

            # Extract header metadata
            headers = {
                "h1": split.metadata.get("h1", ""),
                "h2": split.metadata.get("h2", ""),
                "h3": split.metadata.get("h3", ""),
                "h4": split.metadata.get("h4", ""),
            }

            chunks.append({
                "text": split.page_content,
                "embedding": embedding,
                "file_name": file_name,
                "document_index": document_index,
                "chunk_index": chunk_index,
                "headers": headers
            })

    return chunks
