from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import get_buffer_string
from langchain_core.messages.utils import trim_messages

# Store all chat sessions
session_store = {}

def get_chat_history(session_id: str):
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]

def generate_unique_sessionID():
    import uuid
    return str(uuid.uuid4())

def create_session_history(session_id, token_counter):
    history = get_chat_history(session_id)

    # ✅ Fix: Always convert messages to string before counting tokens
    def safe_token_counter(messages):
        from langchain_core.messages import BaseMessage
        if isinstance(messages, list):
            # Convert list of messages to one string
            text = get_buffer_string(messages)
            return token_counter(text)
        elif isinstance(messages, BaseMessage):
            return token_counter(messages.content)
        else:
            return token_counter(str(messages))

    trimmed_messages = trim_messages(
        history.messages,
        token_counter=safe_token_counter,
        max_tokens=1000
    )

    history.messages = trimmed_messages
    return history
