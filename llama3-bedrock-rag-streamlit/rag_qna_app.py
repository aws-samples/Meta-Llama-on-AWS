import streamlit as st #all streamlit commands will be available through the "st" alias
import rag_llama31_qna as glib #reference to local lib script


st.set_page_config(page_title="Amazon Shareholder RAG QnA Chatbot - powered by Amazon Bedrock Llama 3.1") #HTML title
st.title("RAG based QnA Chatbot") #page title
st.subheader("(Powered by Amazon Bedrock Knowledgebases, Llama 3.1)") #page title

message = st.chat_message("assistant")
message.write("Hello 👋 I am a friendly chat bot who can help you answering questions related to Amazon Shareholder information.")

if 'chat_history' not in st.session_state: #see if the chat history hasn't been created yet
    st.session_state.chat_history = [] #initialize the chat history



chat_container = st.container()

input_text = st.chat_input("Type your question here...") #display a chat input box

if input_text:
    glib.converse_with_model(message_history=st.session_state.chat_history, new_text=input_text)



#Re-render the chat history (Streamlit re-runs this script, so need this to preserve previous chat messages)
for message in st.session_state.chat_history: #loop through the chat history
    with chat_container.chat_message(message.role): #renders a chat line for the given role, containing everything in the with block
        st.markdown(message.text) #display the chat content

