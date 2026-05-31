import asyncio
import json
import os

import httpx
from pydantic import ValidationError

from titan.agents import Agent

LLM_URL = os.getenv("TITAN_LLM_URL", "http://localhost:8001/v1")


async def _stream_and_collect(client: httpx.AsyncClient, payload: dict):
    """POST + stream chunks. Yields events to caller, then yields ('_final', message)"""
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: dict[int, dict] = {}

    async with client.stream(
        "POST", f"{LLM_URL}/chat/completions", json=payload
    ) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line or not line.startswith("data: "):
                continue

            data = line[6:]
            if data == "[DONE]":
                break

            chunk = json.loads(data)
            choice = chunk["choices"][0]
            delta = choice.get("delta", {})

            # Token content
            if text := delta.get("content"):
                content_parts.append(text)
                yield ("token", text)

            # Reasoning content (llama-server extension)
            if reasoning := delta.get("reasoning_content"):
                reasoning_parts.append(reasoning)
                yield ("reasoning", reasoning)

            # Tool call deltas - accumulate by index
            for toolcall_delta in delta.get("tool_calls", []):
                index = toolcall_delta["index"]
                if index not in tool_calls:
                    tool_calls[index] = {
                        "id": "",
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    }

                slot = tool_calls[index]
                if toolcall_delta.get("id"):
                    slot["id"] = toolcall_delta["id"]
                if fn := toolcall_delta.get("function"):
                    if fn.get("name"):
                        slot["function"]["name"] += fn["name"]
                    if fn.get("arguments"):
                        slot["function"]["arguments"] += fn["arguments"]

    # Assemble final message, yield as a sentinel event so caller can pick it up
    message = {"role": "assistant", "content": "".join(content_parts) or None}
    if reasoning_parts:
        message["reasoning_content"] = "".join(reasoning_parts)
    if tool_calls:
        message["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls.keys())]
    yield ("_final", message)


async def run_turn(agent: Agent, conversation: list[dict]):
    """One user message → final assistant message.
    Async generator yielding events for the TUI:
        ("token", str)
        ("reasoning", str)
        ("tool_call", name, args_dict)
        ("tool_result", name, result_str)
    """
    async with httpx.AsyncClient(timeout=None) as client:
        while True:
            payload = {
                "model": agent.model_name,
                "messages": [{"role": "system", "content": agent.system_prompt}]
                + conversation,
                "tools": agent.tool_schemas,
                "temperature": 0,
                "stream": True,
            }

            assistant_msg = None
            async for event in _stream_and_collect(client, payload):
                if event[0] == "_final":
                    assistant_msg = event[1]
                else:
                    yield event

            assert assistant_msg is not None, "Stream ended without _final event"
            conversation.append(assistant_msg)
            if not assistant_msg.get("tool_calls"):
                return  # final response, agent loop done

            # Execute each tool call, validating args with Pydantic before invocation
            for tool_call in assistant_msg["tool_calls"]:
                name = tool_call["function"]["name"]
                raw_args = json.loads(tool_call["function"]["arguments"])
                try:
                    args = agent.tool_input_models[name].model_validate(raw_args)
                except ValidationError as e:
                    result = f"Invalid arguments for {name}: {e}"
                    yield ("tool_result", name, result)
                else:
                    yield ("tool_call", name, raw_args)
                    result = await asyncio.to_thread(agent.tool_funcs[name], args)
                    yield ("tool_result", name, result)
                conversation.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": str(result),
                    }
                )
