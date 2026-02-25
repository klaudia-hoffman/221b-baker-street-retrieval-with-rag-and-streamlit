from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from data_processing import (
    apply_semantic_chunking,
    load_book_with_metadata_by_chapter,
    load_vector_store,
    retrieve_sample_data,
)


def data_processing_and_vector_store_creation():
    # Txt files not added to the repo
    # the data was sourced from Guthenberg project and the specific how/why will be described in a Medium article
    book_as_document_1 = load_book_with_metadata_by_chapter(
        "books/hound-of-the-baskervilles.txt"
    )
    book_as_document_2 = load_book_with_metadata_by_chapter(
        "books/sign-of-the-four.txt"
    )
    book_as_document_3 = load_book_with_metadata_by_chapter(
        "books/study-in-scarlet.txt"
    )
    book_as_document_4 = load_book_with_metadata_by_chapter("books/valley-of-fear.txt")

    combined_documents = (
        book_as_document_1
        + book_as_document_2
        + book_as_document_3
        + book_as_document_4
    )

    semantically_chunked_document = apply_semantic_chunking(combined_documents)
    embeddings = OpenAIEmbeddings()

    vector_store = Chroma.from_documents(
        documents=semantically_chunked_document,
        embedding=embeddings,
        collection_name="valley_of_vectors",
        persist_directory="./chroma_db",
    )

    return vector_store


if __name__ == "__main__":
    # the database creation/adding documents should only be added once
    # if you ran that script multiple times it will append to the database instead of overriding it
    # vector_store = data_processing_and_vector_store_creation()
    vector_store = load_vector_store()
    retrieve_sample_data(vector_store)

# !!!
# to run the streamlit app, please run:
# streamlit run rag_app.py
