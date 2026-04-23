# Edge State Machine Spec

## Purpose

This document defines the edge-side false-positive suppression strategy for door-state detection.

Target:
- reduce `open` false positives
- avoid using raw detector output as a control signal
- provide a stable state output for downstream edge control logic

Scope:
- detector output filtering
- temporal validation
- state machine transition logic
- output contract for edge control integration

## System Layers

The edge pipeline is split into 4 layers:

1. `Detector`
   - runs YOLO and outputs raw detections

2. `Post-Filter`
   - filters invalid or weak detections on a single frame

3. `Temporal Validator`
   - checks persistence across multiple frames

4. `State Machine`
   - converts validated observations into stable state outputs

## Detector Contract

Raw detector output per frame:

```json
{
  "frame_id": 1234,
  "timestamp_ms": 41233,
  "detections": [
    {
      "class_id": 0,
      "class_name": "open",
      "confidence": 0.83,
      "bbox_xyxy": [100, 120, 220, 360]
    },
    {
      "class_id": 1,
      "class_name": "close",
      "confidence": 0.61,
      "bbox_xyxy": [102, 118, 218, 355]
    }
  ]
}
```

Class mapping:
- `0 = open`
- `1 = close`

## Post-Filter Rules

Raw detections must pass all applicable filters before entering temporal validation.

### Threshold Rules

- `open_conf >= OPEN_CONF_THRESHOLD`
- `close_conf >= CLOSE_CONF_THRESHOLD`

Recommended initial values:
- `OPEN_CONF_THRESHOLD = 0.80`
- `CLOSE_CONF_THRESHOLD = 0.55`

### Geometry Rules

- reject boxes with area ratio below `MIN_BOX_AREA_RATIO`
- reject boxes outside configured ROI
- reject boxes with invalid aspect ratio

Recommended initial values:
- `MIN_BOX_AREA_RATIO = 0.010`
- `MIN_ASPECT_RATIO = 0.20`
- `MAX_ASPECT_RATIO = 2.50`

### Consistency Rules

- if the candidate box center jumps too far from recent matched boxes, mark as unstable
- if IoU with recent matched box is too low, reduce trust

Recommended initial values:
- `TRACK_MATCH_IOU = 0.50`
- `MAX_CENTER_SHIFT_RATIO = 0.15`

## Temporal Validator

Single-frame valid detections are still only candidates.

Temporal validation promotes them into confirmed observations.

### Validation Rules

Use `N-of-M persistence`:
- maintain a rolling window of the last `M` frames
- confirm class only if at least `N` frames contain a valid candidate

Recommended initial values:
- `PERSIST_WINDOW = 5`
- `PERSIST_REQUIRED = 3`

### Stability Rules

For the same class:
- matched boxes should keep IoU above threshold or center displacement within limit
- otherwise do not confirm

### Candidate Outputs

Temporal validator outputs:
- `open_candidate_confirmed`
- `close_candidate_confirmed`

It does not directly control downstream behavior.

## State Machine

The state machine is the only component allowed to emit stable door states to edge control.

### States

- `UNKNOWN`
- `CLOSED`
- `OPENING_CANDIDATE`
- `OPEN`
- `CLOSING_CANDIDATE`

### State Diagram

```text
UNKNOWN
  | close confirmed
  v
CLOSED
  | open candidate persists
  v
OPENING_CANDIDATE
  | open confirmed for N frames
  v
OPEN
  | close candidate persists
  v
CLOSING_CANDIDATE
  | close confirmed for N frames
  v
CLOSED
```

Fallback transitions:
- any state can return to `UNKNOWN` after timeout without stable detections

### Transition Rules

`UNKNOWN -> CLOSED`
- if `close_candidate_confirmed` holds for `CLOSE_ENTER_FRAMES`

`CLOSED -> OPENING_CANDIDATE`
- if `open_candidate_confirmed` appears

`OPENING_CANDIDATE -> OPEN`
- if `open_candidate_confirmed` persists for `OPEN_ENTER_FRAMES`

`OPEN -> CLOSING_CANDIDATE`
- if `close_candidate_confirmed` appears or `open` disappears while `close` is dominant

`CLOSING_CANDIDATE -> CLOSED`
- if `close_candidate_confirmed` persists for `CLOSE_ENTER_FRAMES`

`OPENING_CANDIDATE -> CLOSED`
- if `open_candidate_confirmed` drops before confirmation

`CLOSING_CANDIDATE -> OPEN`
- if `open_candidate_confirmed` reappears and dominates

## Hysteresis

Entry and exit rules must not be symmetric.

Purpose:
- avoid flicker
- avoid rapid state toggling

Recommended initial values:
- `OPEN_ENTER_FRAMES = 3`
- `OPEN_EXIT_FRAMES = 5`
- `CLOSE_ENTER_FRAMES = 3`

## Cooldown And Debounce

### Cooldown

After a stable state transition:
- ignore further state transitions for `STATE_COOLDOWN_MS`

Recommended initial value:
- `STATE_COOLDOWN_MS = 2000`

### Debounce

Do not emit a control event on first confirmation.
Only emit after confirmation plus cooldown validation.

## Timeout

If no stable evidence exists for a period:
- move state to `UNKNOWN`

Recommended initial value:
- `STATE_TIMEOUT_MS = 3000`

## ROI Policy

If the camera is fixed, ROI filtering is strongly recommended.

Each deployment site should configure:
- `door_valid_roi`
- optional exclusion zones for reflections or glass areas

Example:

```json
{
  "roi_polygon": [[120, 80], [540, 80], [620, 420], [80, 420]]
}
```

## Parameters

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `OPEN_CONF_THRESHOLD` | `0.80` | Minimum confidence for open candidates |
| `CLOSE_CONF_THRESHOLD` | `0.55` | Minimum confidence for close candidates |
| `MIN_BOX_AREA_RATIO` | `0.010` | Reject too-small detections |
| `MIN_ASPECT_RATIO` | `0.20` | Minimum allowed bbox aspect ratio |
| `MAX_ASPECT_RATIO` | `2.50` | Maximum allowed bbox aspect ratio |
| `TRACK_MATCH_IOU` | `0.50` | IoU threshold for temporal matching |
| `MAX_CENTER_SHIFT_RATIO` | `0.15` | Maximum allowed center drift |
| `PERSIST_WINDOW` | `5` | Rolling validation window length |
| `PERSIST_REQUIRED` | `3` | Required hits inside the window |
| `OPEN_ENTER_FRAMES` | `3` | Frames required to enter `OPEN` |
| `OPEN_EXIT_FRAMES` | `5` | Frames required to leave `OPEN` |
| `CLOSE_ENTER_FRAMES` | `3` | Frames required to enter `CLOSED` |
| `STATE_COOLDOWN_MS` | `2000` | Cooldown after state transition |
| `STATE_TIMEOUT_MS` | `3000` | Timeout to fall back to `UNKNOWN` |

## Input Fields

Per frame input:

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `frame_id` | `int` | yes | Monotonic frame number |
| `timestamp_ms` | `int` | yes | Capture or inference timestamp |
| `detections[].class_id` | `int` | yes | `0=open`, `1=close` |
| `detections[].confidence` | `float` | yes | Detector confidence |
| `detections[].bbox_xyxy` | `list[float]` | yes | Pixel coordinates |
| `camera_id` | `str` | optional | For multi-camera edge nodes |

## Output Fields

Stable edge output:

```json
{
  "state": "OPEN",
  "confidence": 0.91,
  "stable_frames": 4,
  "bbox_xyxy": [100, 120, 220, 360],
  "timestamp_ms": 41233,
  "cooldown_active": false,
  "source_class": "open"
}
```

Output fields:

| Field | Type | Notes |
| --- | --- | --- |
| `state` | `str` | `UNKNOWN`, `CLOSED`, `OPEN` |
| `confidence` | `float` | Aggregated confidence, not single-frame raw conf |
| `stable_frames` | `int` | Number of supporting frames |
| `bbox_xyxy` | `list[float]` | Representative stable bbox |
| `timestamp_ms` | `int` | Timestamp of emitted state |
| `cooldown_active` | `bool` | Whether transition cooldown is active |
| `source_class` | `str` | Dominant detector class behind the state |

## Event Policy

Recommended control events:
- emit only on stable transitions
- do not emit repeated `OPEN` every frame

Suggested events:
- `STATE_CHANGED_OPEN`
- `STATE_CHANGED_CLOSED`
- `STATE_TIMEOUT_UNKNOWN`

## Logging And Replay

For every rejected or unstable edge candidate, keep:
- frame id
- timestamp
- raw detections
- filtered detections
- current state
- reason for rejection

Suggested rejection reasons:
- `below_open_threshold`
- `below_close_threshold`
- `outside_roi`
- `small_box`
- `temporal_not_stable`
- `cooldown_blocked`
- `state_transition_denied`

This log should become the source of future `open_fp_background` and `close_fp_background` mining.

## Pseudocode

```python
state = "UNKNOWN"
cooldown_until = 0
history = RollingWindow(size=5)

def process_frame(frame_id, timestamp_ms, detections):
    filtered = []

    for det in detections:
        if det.class_id == OPEN and det.confidence < OPEN_CONF_THRESHOLD:
            continue
        if det.class_id == CLOSE and det.confidence < CLOSE_CONF_THRESHOLD:
            continue
        if not inside_roi(det.bbox_xyxy):
            continue
        if area_ratio(det.bbox_xyxy) < MIN_BOX_AREA_RATIO:
            continue
        if not valid_aspect_ratio(det.bbox_xyxy):
            continue
        filtered.append(det)

    matched = temporal_match(filtered, history)
    history.append(matched)

    open_confirmed = count_recent(history, "open") >= PERSIST_REQUIRED
    close_confirmed = count_recent(history, "close") >= PERSIST_REQUIRED

    if timestamp_ms < cooldown_until:
        return current_output(state, cooldown_active=True)

    global state, cooldown_until

    if state == "UNKNOWN":
        if close_confirmed:
            state = "CLOSED"
            cooldown_until = timestamp_ms + STATE_COOLDOWN_MS

    elif state == "CLOSED":
        if open_confirmed:
            state = "OPENING_CANDIDATE"

    elif state == "OPENING_CANDIDATE":
        if sustained_open(history, OPEN_ENTER_FRAMES):
            state = "OPEN"
            cooldown_until = timestamp_ms + STATE_COOLDOWN_MS
        elif not open_confirmed:
            state = "CLOSED"

    elif state == "OPEN":
        if sustained_close(history, CLOSE_ENTER_FRAMES):
            state = "CLOSING_CANDIDATE"

    elif state == "CLOSING_CANDIDATE":
        if sustained_close(history, CLOSE_ENTER_FRAMES):
            state = "CLOSED"
            cooldown_until = timestamp_ms + STATE_COOLDOWN_MS
        elif open_confirmed:
            state = "OPEN"

    if stale(timestamp_ms, history, STATE_TIMEOUT_MS):
        state = "UNKNOWN"

    return current_output(state, cooldown_active=False)
```

## Deployment Notes

- for edge deployment, detector output should be treated as a noisy sensor
- stable control behavior comes from the state machine, not from YOLO alone
- threshold tuning should be done per camera and per scene
- ROI configuration should be site-specific

## Recommended First Implementation

Implement the following first:
- class-specific thresholds
- ROI filtering
- `3-of-5` temporal persistence
- `OPEN/CLOSED/UNKNOWN` state machine
- `2s` cooldown

This is the minimum practical version for reducing `open` flicker false positives on edge.
