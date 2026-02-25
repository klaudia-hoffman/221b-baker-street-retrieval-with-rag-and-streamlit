import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

import data_processing

# Streamlit UI top of the page set up
st.set_page_config(page_title="RAG with Citations", layout="centered")
st.title("📚 Book Assistant with Citations")
st.text(
    "Ask a question, and you will receive an answer complete with exact source citations. Currently, this tool searches"
    " across Arthur Conan Doyle's four full-length Sherlock Holmes novels: A Study in Scarlet, The Sign of the Four,"
    " The Hound of the Baskervilles, and The Valley of Fear."
)


@st.cache_resource
def load_vector_db():
    return data_processing.load_vector_store()


def format_docs_with_citations(docs):
    formatted = []
    for i, doc in enumerate(docs, 1):
        metadata_info = f"[{i}] {doc.metadata.get('source', 'Unknown source')}"
        metadata_info += f", Author: {doc.metadata.get('author', 'Unknown author')}"
        metadata_info += f", chapter: {doc.metadata.get('chapter', 'Unknown')}"
        metadata_info += (
            f", Chapter Title: {doc.metadata.get('chapter_title', 'Unknown')}"
        )
        formatted.append(f"{metadata_info}\n{doc.page_content}")
    return "\n\n".join(formatted)


def get_documents_with_citations(vector_store, question):
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    retriever = vector_store.as_retriever(search_kwargs={"k": 10})

    # Source attribution prompt template
    with_citation_prompt = ChatPromptTemplate.from_template("""
    You are a precise research assistant. Your sole job is to answer questions using ONLY the sources provided — do not use prior knowledge.
    
    ## Rules
    - Cite every factual claim inline using [1], [2], etc., immediately after the claim.
    - If multiple sources support the same claim, cite all of them: [1][2].
    - If the sources do not contain enough information to answer the question, respond with: "The provided sources do not contain sufficient information to answer this question."
    - Do not speculate, infer, or add information beyond what is explicitly stated in the sources.
    - Keep your answer concise and direct.
    
    ## Question
    {question}
    
    ## Sources
    {sources}
    
    ## Answer
    Provide your answer here, with inline citations. End with a "References" section listing each cited source by number.
    """)

    retrieved_docs = retriever.invoke(question)

    formatted_docs = format_docs_with_citations(retrieved_docs)

    citation_chain = with_citation_prompt | llm | StrOutputParser()

    chain_response = citation_chain.invoke(
        {
            "question": question,
            "sources": formatted_docs,
        }
    )

    return chain_response, retrieved_docs


# ---
#  UI with streamlit
SAMPLE_QUESTIONS = [
    "Where does Sherlock live?",
    "Who is Sherlock Holmes's ultimate intellectual rival?",
    "What is 'arga treasure'?",
    "What happened in Dartmoor?",
]

with st.expander("💡 Sample questions", expanded=False):
    for q in SAMPLE_QUESTIONS:
        if st.button(q, key=q):
            st.session_state["question_input"] = q


question = st.text_area(
    "Your question",
    value=st.session_state.get("question_input", ""),
    placeholder="Ask something about Sherlock Holmes novels...",
    height=100,
    key="question_input",
)

vector_store = load_vector_db()

ask_clicked = st.button("Ask", disabled=not question.strip())

if ask_clicked:
    with st.spinner("Retrieving sources and generating answer..."):
        try:
            answers, documents = get_documents_with_citations(vector_store, question)
            # Answer card
            st.markdown(
                f'<div class="answer-card">{answers.replace(chr(10), "<br>")}</div>',
                unsafe_allow_html=True,
            )

            # Retrieved sources
            st.markdown("<div style='margin-top:1.5rem'>", unsafe_allow_html=True)
            st.markdown("**Retrieved sources**")
            for i, doc in enumerate(documents, 1):
                src = doc.metadata.get("source", "Unknown")
                author = doc.metadata.get("author", "")
                chapter = doc.metadata.get("chapter", "")
                chapter_title = doc.metadata.get("chapter_title", "")
                label = (
                    f"[{i}] {src}"
                    + (f",  Author {author}" if author else "")
                    + (f",  Chapter: {chapter}" if chapter else "")
                    + (f",  Chapter title: {chapter_title}" if chapter_title else "")
                )
                with st.expander(label):
                    st.markdown(
                        f"<span style='color:#a09080;font-size:0.9rem'>{doc.page_content}</span>",
                        unsafe_allow_html=True,
                    )
            st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error: {e}")
