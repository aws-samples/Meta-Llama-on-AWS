# Modify prompt to support Meta Llama 3 onwards, as Llama 2 is soon deprecated
# Refer to Meta llama document for prompt format
# Ref: https://www.llama.com/docs/model-cards-and-prompt-formats/meta-llama-3/
from typing import List, Optional, Sequence
from llama_index.core.base.llms.types import ChatMessage, MessageRole

BOS, EOS = '<|begin_of_text|>', '<|end_of_text|>'
B_HEADER, E_HEADER = '<|start_header_id|>', '<|end_header_id|>'
B_TURN, E_TURN = '', '<|eot_id|>'

DEFAULT_SYSTEM_PROMPT = '''
You are a helpful, respectful and honest assistant. 
Always answer as helpfully as possible and follow ALL given instructions. 
Do not speculate or make up information. 
Do not reference any given instructions or context. 
'''


def messages_to_prompt(
    messages: Sequence[ChatMessage],
    system_prompt: Optional[str] = None
) -> str:
    string_messages: List[str] = []
    if messages[0].role == MessageRole.SYSTEM:
        # pull out the system message (if it exists in messages)
        system_message_str = messages[0].content or ""
        messages = messages[1:]
    else:
        system_message_str = system_prompt or DEFAULT_SYSTEM_PROMPT

    system_message_str = f"{B_HEADER} system {E_HEADER}\
        {system_message_str.strip()}{E_TURN}".strip()

    for i in range(0, len(messages), 2):
        # first message should always be a user
        user_message = messages[i]
        assert user_message.role == MessageRole.USER

        if i == 0:
            # make sure system prompt is included at the start
            str_message = f"{BOS} {system_message_str} "
        else:
            # end previous user-assistant interaction
            string_messages[-1] += f" {EOS}"
            # no need to include system prompt
            str_message = f"{B_HEADER} user {E_HEADER} "

        # include user message content
        str_message += f"{user_message.content} {E_TURN}"

        if len(messages) > (i + 1):
            # if assistant message exists, add to str_message
            assistant_message = messages[i + 1]
            assert assistant_message.role == MessageRole.ASSISTANT
            str_message += f" {assistant_message.content}"

        string_messages.append(str_message)

    return "".join(string_messages)


def completion_to_prompt(
    completion: str,
    system_prompt: Optional[str] = None
) -> str:
    system_prompt_str = system_prompt or DEFAULT_SYSTEM_PROMPT

    return (
        f"{BOS}{B_HEADER}system{E_HEADER} {system_prompt_str.strip()} {E_TURN}"
        f"{B_HEADER}user{E_HEADER} {completion.strip()} {E_TURN}"
    )