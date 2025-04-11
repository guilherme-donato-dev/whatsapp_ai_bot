from config import REDIS_URL
from langchain_community.chat_message_histories import RedisChatMessageHistory


def get_session_history(session_id):
    return RedisChatMessageHistory(
        session_id=session_id,
        url=REDIS_URL,
    )


