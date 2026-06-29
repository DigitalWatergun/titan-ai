from titan.agents import MAIN_AGENT
from titan.loop import complete_turn

_SUMMARY_SYSTEM = """Update the running summary of this chat. Merge the new messages into the existing summary; keep it concise but preserve decisions, file paths, and open threads. Reply with the summary only."""

_SUMMARY_TIMEOUT = 300


async def _summarize(old_summary: str, delta: list[dict]) -> str:
    delta_text = "\n".join(f"{m['role']}: {m.get('content', '')}" for m in delta)
    user = f"Existing summary:\n{old_summary or '(none yet)'}\n\nNew messages:\n{delta_text}"
    return await complete_turn(
        [
            {"role": "system", "content": _SUMMARY_SYSTEM},
            {"role": "user", "content": user},
        ],
        model=MAIN_AGENT.model_name,
        timeout=_SUMMARY_TIMEOUT,
    )


def _snap_to_turn_boundary(archive: list[dict], cutoff: int) -> int:
    for i in range(min(cutoff, len(archive) - 1), -1, -1):
        if archive[i].get("role") == "user":
            return i
    return 0


async def compact(store, keep_recent_n: int = 6) -> tuple[list[dict], int, str] | None:
    archive = store.messages
    raw_cutoff = len(archive) - keep_recent_n
    new_covers_through = _snap_to_turn_boundary(archive, raw_cutoff)
    prev = store.compaction
    old_covers_through = prev["covers_through"]
    old_summary = prev["summary"]
    if new_covers_through <= old_covers_through:
        return None
    delta = archive[old_covers_through:new_covers_through]
    summary_text = await _summarize(old_summary, delta)
    return delta, new_covers_through, summary_text
