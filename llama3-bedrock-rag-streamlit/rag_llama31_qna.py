import json
import boto3
from botocore.client import Config
from langchain.prompts import PromptTemplate

MAX_MESSAGES = 20
MODEL_ID = 'meta.llama3-1-70b-instruct-v1:0'
KNOWLEDGE_BASE_ID = "DYTL71ODQZ"

bedrock_client = boto3.client(service_name='bedrock-runtime')

class ChatMessage(): 
    def __init__(self, role, text):
        self.role = role
        self.text = text

def get_tools():
    tools = [
        {
                "toolSpec": {
                    "name": "amazon_shareholder_information",
                    "description": "Retrieve information about Amazon shareholder 2023 documents.",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The retrieval augmented generation query used to search information in the knowledgebase about Amazon shareholder info."
                                }
                            },
                            "required": [
                                "query"
                            ]
                        }
                    }
                }
        }
    ]

    return tools

def transform_messages_for_api(chat_messages):
    return [{"role": msg.role, "content": [{"text": msg.text}]} for msg in chat_messages]
    
def convert_chat_messages_to_converse_api(chat_messages):
    messages = []

    for chat_msg in chat_messages:
        messages.append({
            "role": chat_msg.role,
            "content": [
                {
                    "text": chat_msg.text
                }
            ]
        })
    return messages

def process_tool(response_message, messages, bedrock, tool_list):
    messages.append(response_message)
    
    response_content_blocks = response_message['content']
    follow_up_content_blocks = []
    
    for content_block in response_content_blocks:
        if 'toolUse' in content_block:
            tool_use_block = content_block['toolUse']
            
            if tool_use_block['name'] == 'amazon_shareholder_information':

                query = tool_use_block['input']['query']
                rag_content = get_shareholder_info(query)
                
                follow_up_content_blocks.append({
                    "toolResult": {
                        "toolUseId": tool_use_block['toolUseId'],
                        "content": [
                            { "text": rag_content }
                        ]
                    }
                })
                
                
    if len(follow_up_content_blocks) > 0:
        
        follow_up_message = {
            "role": "user",
            "content": follow_up_content_blocks,
        }
    
        messages.append(follow_up_message)
        
        response = bedrock.converse(
            modelId=MODEL_ID,
            messages=messages,
            inferenceConfig={
                "maxTokens": 2000,
                "temperature": 0,
                "topP": 0.9,
                "stopSequences": []
            },
            toolConfig={
                "tools": tool_list
            }
        )
            
        return True, response['output']['message']['content'][0]['text'] 
        
    else:
        return False, None 

def get_contexts(retrievalResults):
    contexts = []
    for retrievedResult in retrievalResults:
        text = retrievedResult['content']['text']
        if text.startswith("Document 1: "):
            text = text[len("Document 1: "):]
        contexts.append(text)
    contexts_string = ', '.join(contexts)
    return contexts_string
    
def get_shareholder_info(question):
    response_retrieve = retrieve(question, KNOWLEDGE_BASE_ID)["retrievalResults"]
    contexts = get_contexts(response_retrieve)

    PROMPT_TEMPLATE = """DOCUMENT:
    {context}               
    QUESTION:
    {message}                  
    INSTRUCTIONS:
    Answer the user's QUESTION using only the DOCUMENT text above. Greet friendly if the QUESTION contains "hi" or "hello"
    Keep your answer strictly grounded in the facts provided. Do not refer to the "DOCUMENT," "documents," "provided text," ,"based on.." or any similar phrases in your answer.
    If the provided text contains the facts to answer the QUESTION, include all relevant details in your answer.
    If the provided text doesn’t contain the facts to answer the QUESTION, respond only with "I don't know" and do not add any further information.
    """

    prompt = PromptTemplate(template=PROMPT_TEMPLATE, 
                               input_variables=["context","message"])

    prompt_final = prompt.format(context=contexts,
                                 message=question)

    native_request = {
        "prompt": prompt_final,
        "max_gen_len": 2048,
        "temperature": 0.5,
    }
    
    # Convert the native request to JSON.
    request = json.dumps(native_request)
    model_id = MODEL_ID
    accept = 'application/json'
    content_type = 'application/json'
    response = bedrock_client.invoke_model(body=request, modelId=model_id, accept=accept, contentType=content_type)
    response_body = json.loads(response.get('body').read())

    if response_body.get('content') and response_body['content'][0].get('text'):
        response_text = response_body['content'][0]['text']
    elif response_body.get('generation'):
        response_text = response_body['generation']
    else:
        response_text = "Sorry, I didn't get it"

    return response_text

def retrieve(query, kbId, numberOfResults=3):
    bedrock_config = Config(connect_timeout=120, read_timeout=120, retries={'max_attempts': 0})

    bedrock_agent_client = boto3.client("bedrock-agent-runtime",config=bedrock_config)
    return bedrock_agent_client.retrieve(
        retrievalQuery= {
            'text': query
        },
        knowledgeBaseId=kbId,
        retrievalConfiguration= {
            'vectorSearchConfiguration': {
                'numberOfResults': numberOfResults,
                'overrideSearchType': "HYBRID"
            }
        }
    )

def converse_with_model(message_history, new_text=None):
    session = boto3.Session()
    bedrock = session.client(service_name='bedrock-runtime')
    
    tool_list = get_tools()
    
    new_text_message = ChatMessage('user', text=new_text)
    message_history.append(new_text_message)
    
    number_of_messages = len(message_history)
    
    if number_of_messages > MAX_MESSAGES:
        del message_history[0 : (number_of_messages - MAX_MESSAGES) * 2] 
    
    messages = transform_messages_for_api(message_history)
    
    response = bedrock.converse(
        modelId=MODEL_ID,
        messages=messages,
        inferenceConfig={
            "maxTokens": 2000,
            "temperature": 0,
            "topP": 0.9,
            "stopSequences": []
        },
        toolConfig={
            "tools": tool_list
        }
    )
    
    response_message = response['output']['message']
    
    tool_used, output = process_tool(response_message, messages, bedrock, tool_list)
    
    if not tool_used:
        output = response['output']['message']['content'][0]['text']
    
   
    response_chat_message = ChatMessage('assistant', output)
    message_history.append(response_chat_message)
    
    return
