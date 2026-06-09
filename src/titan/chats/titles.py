from titan.loop import complete_turn

_TITLE_SYSTEM = (
    "Generate a concise 3-6 word title for this chat. "
    "Reply with the title only — no quotes, no punctuation."
)


async def generate_title(messages: list[dict], model: str) -> str:
    first_user = next(
        (m["content"] for m in messages if m["role"] == "user" and m.get("content")), ""
    )
    if not first_user:
        return "Untitled"
    try:
        title = await complete_turn(
            [
                {"role": "system", "content": _TITLE_SYSTEM},
                {"role": "user", "content": first_user},
            ],
            model=model,
        )
        return title.strip().strip('"')[:50] or _derive_title(messages)
    except Exception:
        return _derive_title(messages)  # fallback to truncated-first-message


def _derive_title(messages: list[dict], max_len: int = 50) -> str:
    for m in messages:
        if m["role"] == "user" and m.get("content"):
            t = " ".join(m["content"].split())
            return t[:max_len] + ("…" if len(t) > max_len else "")
    return "Untitled"
