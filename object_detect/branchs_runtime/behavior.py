from __future__ import annotations

from dataclasses import dataclass, field

from .tracking import Track


@dataclass(slots=True)
class TrackBehaviorState:
    stable_label: str = "unknown"
    open_streak: int = 0
    close_streak: int = 0
    transition_trace: list[str] = field(default_factory=list)


class BehaviorAgent:
    def __init__(self, min_transition_hits: int = 2) -> None:
        self.min_transition_hits = max(1, min_transition_hits)
        self._states: dict[int, TrackBehaviorState] = {}

    def update(self, track: Track) -> dict[str, object]:
        state = self._states.setdefault(track.track_id, TrackBehaviorState())
        label = track.secondary_states.get("door_gap_state", track.cls_label)

        if label == "open":
            state.open_streak += 1
            state.close_streak = 0
        elif label == "close":
            state.close_streak += 1
            state.open_streak = 0
        else:
            state.open_streak = 0
            state.close_streak = 0

        event_name = "idle"
        state_flag = "Ongoing"
        confidence = max(track.det_conf, track.cls_conf)
        debug_reason = (
            f"label={label}, stable={state.stable_label}, "
            f"open_streak={state.open_streak}, close_streak={state.close_streak}"
        )

        if label == "open" and state.stable_label != "open" and state.open_streak >= self.min_transition_hits:
            previous = state.stable_label
            state.stable_label = "open"
            event_name = "door_opening"
            state_flag = "Start"
            state.transition_trace.append(f"{previous}->open")
            debug_reason = f"{debug_reason}, transition={previous}->open"
        elif label == "close" and state.stable_label != "close" and state.close_streak >= self.min_transition_hits:
            previous = state.stable_label
            state.stable_label = "close"
            event_name = "door_closing"
            state_flag = "Start"
            state.transition_trace.append(f"{previous}->close")
            debug_reason = f"{debug_reason}, transition={previous}->close"
        elif state.stable_label == "open":
            event_name = "door_open"
        elif state.stable_label == "close":
            event_name = "door_closed"

        return {
            "track_id": track.track_id,
            "bbox": track.bbox,
            "secondary_states": dict(track.secondary_states),
            "event_name": event_name,
            "state_flag": state_flag,
            "confidence": round(confidence, 4),
            "debug_reason": debug_reason,
            "transition_trace": list(state.transition_trace),
        }
