import streamlit as st
from pdf_processor import extract_pdf_text
from rag_pipeline import create_chunks
from vector_store import create_vector_store, search_chunks
from llm_service import generate_answer

st.title("Insurance Policy AI")

# Initialize session state variables so they persist across reruns
if "vector_index" not in st.session_state:
    st.session_state.vector_index = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "processed_file_name" not in st.session_state:
    st.session_state.processed_file_name = None

policy_file = st.file_uploader(
    "Upload Insurance Policy PDF",
    type=["pdf"]
)

## TODO
# estimate_file = st.file_uploader(
#     "Upload Hospital Estimate PDF (Optional)",
#     type=["pdf"]
# )

question = st.text_input(
    "Ask a question about your policy"
)

# Phase 1 & 2: Process PDF and build vector store ONLY if a new file is uploaded
if policy_file:
    # Check if this is a newly uploaded file or if we already processed it
    if st.session_state.processed_file_name != policy_file.name:
        with st.spinner("Processing policy and generating embeddings..."):
            # Extract text
            data = extract_pdf_text(policy_file)
            
            # Create chunks
            chunks = create_chunks(data["text"])
            
            # Create vector store
            vector_index, _ = create_vector_store(chunks)
            
            # Save to session state to prevent reprocessing on next rerun
            st.session_state.vector_index = vector_index
            st.session_state.chunks = chunks
            st.session_state.processed_file_name = policy_file.name
            
        st.success("Policy uploaded and indexed successfully!")
else:
    # Clear the session state if the user removes the file
    st.session_state.vector_index = None
    st.session_state.chunks = None
    st.session_state.processed_file_name = None

# Phase 3: Query handling
if st.button("Ask AI"):
    # Input validation guardrails
    if not policy_file:
        st.error("Please upload an insurance policy PDF first.")
    elif not question.strip():
        st.warning("Please enter a question.")
    elif st.session_state.vector_index is not None and st.session_state.chunks is not None:
        
        with st.spinner("Searching policy and generating answer..."):
            # Search using cached index and chunks
            results = search_chunks(
                question,
                st.session_state.vector_index,
                st.session_state.chunks,
                top_k=3
            )

            context = "\n\n".join(results)

            answer = generate_answer(
                context,
                question
            )

        # Display Chat UI
        st.subheader("AI Answer")
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            st.write(answer)

        # Display Sources
        st.subheader("Sources Used")
        for i, chunk in enumerate(results):
            with st.expander(f"Source {i+1}"):
                st.write(chunk)
