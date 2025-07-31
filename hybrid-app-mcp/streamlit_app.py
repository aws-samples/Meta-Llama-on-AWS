# streamlit_app.py
import streamlit as st
import requests
import json

st.set_page_config(page_title="PDF Document Analyzer", layout="wide")
st.title("PDF Document Analyzer")
st.subheader("Powered by Meta Llama 3.2 11B & AWS")

# Add server status check in sidebar
try:
    health_response = requests.get("http://localhost:5000/health")
    if health_response.status_code == 200:
        st.sidebar.success("✅ Server is running")
    else:
        st.sidebar.error("❌ Server is not responding properly")
except:
    st.sidebar.error("❌ Cannot connect to server")

# Main interface
col1, col2 = st.columns(2)

with col1:
    input_type = st.radio("Choose input type:", ["URL", "Local File"])

    if input_type == "URL":
        source = st.text_input("Enter PDF URL:")
        is_url = True
    else:
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        source = uploaded_file.name if uploaded_file else None
        is_url = False

with col2:
    question = st.text_area("What would you like to know about the document?",
                           height=100)

if st.button("Analyze Document"):
    if source and question:
        with st.spinner('Analyzing document...'):
            try:
                response = requests.post(
                    "http://localhost:5000/analyze",
                    json={
                        "source": source,
                        "question": question,
                        "is_url": is_url
                    }
                )

                if response.status_code == 200:
                    result = response.json()
                    st.success("Analysis complete!")
                    st.write("Results:")
                    st.write(result['analysis'])
                else:
                    st.error(f"Error: {response.json().get('error', 'Unknown error')}")

            except Exception as e:
                st.error(f"Error connecting to server: {str(e)}")
    else:
        st.warning("Please provide both a document and a question.")