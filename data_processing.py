import json
import re
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.load import dump, load
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings


def load_book_with_metadata_by_chapter(file_path: str) -> List[Document]:
    documents = []
    chapter_patters = re.compile(r"CHAPTER \d+")
    chapter_title_patters = re.compile(r"C NAME:\s+(.+)$")
    title_patters = re.compile(r"TITLE:\s+(.+)$")
    author_patters = re.compile(r"AUTHOR:\s+(.+)$")

    current_chapter = "Introduction"
    combined_chapter = ""
    chapter_number = 0
    chapter_title = ""
    book_title = ""
    book_author = ""
    with open(file_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped_line = line.strip()
            # get additional metadata
            if title_patters.match(stripped_line):
                book_title = stripped_line.removeprefix("TITLE: ")
            if author_patters.match(stripped_line):
                book_author = stripped_line.removeprefix("AUTHOR: ")
            if chapter_title_patters.match(stripped_line):
                chapter_title = stripped_line.removeprefix("C NAME: ")

            if not stripped_line:  # skip empty lines
                continue
            if chapter_patters.match(
                stripped_line
            ):  # check if line looks like new chapter
                # then first save the old chapter title and all the content to a document, then update the chapter name
                if chapter_number > 0:
                    # create the document for this line
                    doc = Document(
                        page_content=combined_chapter,
                        metadata={
                            "source": book_title,
                            "author": book_author,
                            "chapter_title": chapter_title,
                            "chapter": current_chapter,
                        },
                    )
                    documents.append(doc)
                    combined_chapter = ""

                current_chapter = stripped_line
                chapter_number += 1
            else:
                combined_chapter = combined_chapter + " " + stripped_line
        if combined_chapter:
            doc = Document(
                page_content=combined_chapter,
                metadata={
                    "source": book_title,
                    "author": book_author,
                    "chapter_title": chapter_title,
                    "chapter": current_chapter,
                },
            )
            documents.append(doc)

    return documents


def save_documents(documents: List[Document], file_path: str):
    with open(file_path, "w") as f:
        # dump converts the list of Documents into a JSON-serializable format
        json.dump(dump.dumpd(documents), f, indent=2)


def load_documents(file_path: str) -> List[Document]:
    with open(file_path, "r") as f:
        data = json.load(f)
        # load reconstructs the actual Document objects
        docs = load(data)
        return docs


def apply_semantic_chunking(documents: List[Document]) -> List[Document]:
    embeddings = OpenAIEmbeddings()
    text_splitter = SemanticChunker(embeddings=embeddings, add_start_index=True)
    documents = text_splitter.transform_documents(documents)
    return documents


def add_processed_documents_to_vector_db(filepath: str):
    documents = load_documents(filepath)
    embeddings = OpenAIEmbeddings()

    # Create the Vector Store (Chroma)
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="valley_of_vectors",
        persist_directory="./chroma_db",
    )
    return vector_store


def load_vector_store():
    # Initialize the same embedding model used during creation
    embeddings = OpenAIEmbeddings()

    # Load the Vector Store from disk
    vector_store = Chroma(
        collection_name="valley_of_vectors",
        persist_directory="./chroma_db",
        embedding_function=embeddings,
    )

    return vector_store


def clean_vector_store():  # in case it got injested twice, it will not overwrite the db, but append to it!
    vector_store = load_vector_store()
    try:
        vector_store.delete_collection()
    except:
        pass


# small function to test retrieval during development
def retrieve_sample_data(vector_store: Chroma):
    # Create the Retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    #  Test It
    print("--- Querying the Database ---")
    results = retriever.invoke(
        "Who is Sherlock Holmes's ultimate intellectual rival?"
    )  # <- change questions here

    for doc in results:
        print(f"Content: {doc.page_content}")
        print(f"Source: {doc.metadata['source']} | Author: {doc.metadata['author']}")
        print(
            f"Chapter: {doc.metadata['chapter']} | Chapter Title: {doc.metadata['chapter_title']}"
        )
        print("-" * 20)


def combine_data_processing_and_save_to_json(book_file_path: str) -> List[Document]:
    list_of_documents = load_book_with_metadata_by_chapter(book_file_path)
    saved_documents_by_chapter = "books_by_chapters.json"
    save_documents(list_of_documents, saved_documents_by_chapter)
    list_of_documents_semantically_chunked = apply_semantic_chunking(list_of_documents)
    saved_documents_chucked = "books_by_chapters_semantically_chunked.json"
    save_documents(list_of_documents_semantically_chunked, saved_documents_chucked)
