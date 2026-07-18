import streamlit as st
from pdf_processor import extract_pdf_text
from rag_pipeline import create_chunks
from vector_store import create_vector_store
from llm_service import generate_answer
from vector_store import search_chunks

st.title("Insurance Policy AI")

policy_file = st.file_uploader(
    "Upload Insurance Policy PDF",
    type=["pdf"]
)

# estimate_file = st.file_uploader(
#     "Upload Hospital Estimate PDF (Optional)",
#     type=["pdf"]
# )

question = st.text_input(
    "Ask a question about your policy"
)

if policy_file:

    data = extract_pdf_text(policy_file)

    chunks = create_chunks(data["text"])

    vector_index, embeddings = create_vector_store(chunks)

    # st.success(
    #     "Embeddings generated successfully"
    # )

    # st.write(
    #     f"Embedding Dimension: {embeddings.shape[1]}"
    # )

    # st.write(
    #     f"Total Embeddings: {embeddings.shape[0]}"
    # )

    # st.write(
    #     f"Total Chunks Created: {len(chunks)}"
    # )

    # st.subheader("First Chunk Preview")

    # st.write(chunks[0][:1000])

    st.success("Policy uploaded successfully")

    # st.write(
    #     f"Pages: {data['pages']}"
    # )

    # st.write(
    #     f"Characters: {len(data['text'])}"
    # )

    # st.subheader(
    #     "Preview"
    # )

    # st.text(
    #     data["text"][:1000]
    # )

if st.button("Ask AI"):

    if policy_file and question:

        results = search_chunks(
            question,
            vector_index,
            chunks,
            top_k=3
        )

        context = "\n\n".join(results)

        answer = generate_answer(
            context,
            question
        )

        st.subheader("AI Answer")

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            st.write(answer)

        # st.write(answer)

        st.subheader("Sources")

        for i, chunk in enumerate(results):
            with st.expander(
                f"Source {i+1}"
            ):
                st.write(chunk)
