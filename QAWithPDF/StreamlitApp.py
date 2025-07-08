import streamlit as st
from QAWithPDF.data_ingestion import load_data
from QAWithPDF.embedding import download_gemini_embedding
from QAWithPDF.model_api import load_model
import tempfile
import os

    
def main():
    st.set_page_config("QA with Documents")
    
    uploaded_file = st.file_uploader("Upload your document", type=["pdf"])
    
    st.header("QA with Documents(Information Retrieval)")
    
    user_question= st.text_input("Ask your question")
    
    if st.button("Submit & Process"):
        if uploaded_file is not None and user_question.strip():
            with st.spinner("Processing..."):
                with tempfile.TemporaryDirectory() as temp_dir:
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.read())
                    documents = load_data(temp_dir)  # Pass the directory, not the file
                    model=load_model()
                    query_engine=download_gemini_embedding(model,documents)
                        
                    response = query_engine.query(user_question)
                        
                    st.write(response.response)
                
                
if __name__=="__main__":
    main()
