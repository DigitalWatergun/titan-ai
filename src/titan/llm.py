from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI


class ChatOpenAIWithReasoning(ChatOpenAI):
    """ChatOpenAI subclass that preserves reasoning_content from the API response.

    LangChain's stock ChatOpenAI silently drops reasoning_content from
    OpenAI-compatible providers (llama.cpp, vLLM, etc). This subclass
    extracts it and stores it in AIMessage.additional_kwargs so the CLI
    can display the model's thinking in real time.
    """

    def _create_chat_result(self, response, generation_info=None):
        result = super()._create_chat_result(response, generation_info)
        try:
            raw_message = response.choices[0].message
            reasoning = getattr(raw_message, "reasoning_content", None)
            if reasoning is None:
                msg_dict = (
                    raw_message.model_dump()
                    if hasattr(raw_message, "model_dump")
                    else {}
                )
                reasoning = msg_dict.get("reasoning_content")
            if reasoning:
                for gen in result.generations:
                    if isinstance(gen.message, AIMessage):
                        gen.message.additional_kwargs["reasoning_content"] = reasoning
        except Exception:
            pass
        return result
