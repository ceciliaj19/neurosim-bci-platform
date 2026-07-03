"""NeuroLab — Closed-Loop BCI page: session-style motor imagery decoder demo."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from components.cursor_chart import render_cursor_chart
from components.exporter import download_buttons
from components.page_header import render_page_header
from components.ui import section_header
from neurosim.bci import CursorController

# ---------------------------------------------------------------------------
# Session state keys (all cl_* namespace)
# ---------------------------------------------------------------------------
_CL_ACTIVE = "cl_active"
_CL_CUES = "cl_cues"
_CL_TRIAL_IDX = "cl_trial_idx"
_CL_HISTORY = "cl_history"
_CL_CURSOR = "cl_cursor"
_CL_TRAJ = "cl_trajectory"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_CUE_ICONS: dict[str, str] = {"left": "←", "right": "→"}
_CUE_COLORS: dict[str, str] = {"left": "#1967d2", "right": "#c5221f"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_predict(cue: str, accuracy: float) -> tuple[str, float]:
    """Return (prediction, confidence) from a probabilistic mock BCI decoder.

    When correct, confidence is drawn from U(0.65, 0.95); when wrong, from
    U(0.50, 0.70) — reflecting that real BCIs are less confident on errors.
    """
    correct = random.random() < accuracy
    prediction = cue if correct else ("right" if cue == "left" else "left")
    confidence = random.uniform(0.65, 0.95) if correct else random.uniform(0.50, 0.70)
    return prediction, round(confidence, 2)


def _generate_cues(n_trials: int, seed: int) -> list[str]:
    """Balanced, shuffled cue list: equal left / right counts."""
    half = n_trials // 2
    remainder = n_trials - 2 * half
    cues = ["left"] * half + ["right"] * half + ["left"] * remainder
    rng = random.Random(seed)
    rng.shuffle(cues)
    return cues


def _start_session(n_trials: int, seed: int) -> None:
    st.session_state[_CL_CUES] = _generate_cues(n_trials, seed)
    st.session_state[_CL_TRIAL_IDX] = 0
    st.session_state[_CL_HISTORY] = []
    st.session_state[_CL_CURSOR] = CursorController(step_size=1.0)
    st.session_state[_CL_TRAJ] = [(0.0, 0.0)]
    st.session_state[_CL_ACTIVE] = True


def _reset_session() -> None:
    for key in (_CL_ACTIVE, _CL_CUES, _CL_TRIAL_IDX, _CL_HISTORY,
                _CL_CURSOR, _CL_TRAJ):
        st.session_state.pop(key, None)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

def render() -> None:
    render_page_header(
        "🎮", "Closed-Loop BCI",
        "Session-style motor imagery decoder demo",
    )

    with st.expander("ℹ️ How closed-loop BCI works"):
        st.markdown(
            "In a **closed-loop BCI**, the participant sees a directional cue "
            "(e.g. *imagine moving your left hand*), brain signal is recorded "
            "and decoded in real time, and the decoded intent drives an output "
            "device — here, a virtual 2D cursor. "
            "The 'closed loop' means the cursor's movement is visible to the "
            "participant, who can adjust their imagery strategy based on the "
            "feedback.\n\n"
            "This demo simulates the decode step with a probabilistic mock "
            "decoder. Set **Decoder Accuracy** in the sidebar to explore how "
            "classifier performance affects session accuracy and cursor trajectory."
        )

    active = st.session_state.get(_CL_ACTIVE, False)

    # -- Sidebar -----------------------------------------------------------------
    with st.sidebar:
        st.markdown("**Session Settings**")
        n_trials = st.slider(
            "Number of trials", 4, 20, 10, 2,
            disabled=active,
            key="cl_n_trials",
        )
        decoder_accuracy = st.slider(
            "Decoder Accuracy (%)", 50, 100, 75, 5,
            disabled=active,
            key="cl_accuracy",
        ) / 100.0
        seed = int(st.number_input(
            "Random seed", min_value=0, max_value=9999, value=42, step=1,
            disabled=active,
            key="cl_seed",
        ))

        st.divider()
        st.markdown("**Controls**")

        if not active:
            if st.button("▶ Start Session", key="cl_start", type="primary",
                         use_container_width=True):
                _start_session(n_trials, seed)
                st.rerun()
        else:
            cues = st.session_state[_CL_CUES]
            trial_idx = st.session_state[_CL_TRIAL_IDX]
            session_complete = trial_idx >= len(cues)

            if not session_complete:
                if st.button("▶ Next Trial", key="cl_next", type="primary",
                             use_container_width=True):
                    cue = cues[trial_idx]
                    pred, conf = _mock_predict(cue, decoder_accuracy)
                    correct = pred == cue
                    pos = st.session_state[_CL_CURSOR].move(pred)
                    st.session_state[_CL_TRAJ].append(pos)
                    st.session_state[_CL_HISTORY].append({
                        "trial": trial_idx + 1,
                        "cue": cue,
                        "prediction": pred,
                        "confidence": conf,
                        "correct": correct,
                    })
                    st.session_state[_CL_TRIAL_IDX] = trial_idx + 1
                    st.rerun()

            if st.button("↺ Reset Session", key="cl_reset",
                         use_container_width=True):
                _reset_session()
                st.rerun()

        # Live progress summary
        if active:
            history = st.session_state.get(_CL_HISTORY, [])
            n_done = len(history)
            n_correct = sum(t["correct"] for t in history)
            cues = st.session_state[_CL_CUES]

            st.divider()
            st.markdown("**Session Progress**")
            st.metric("Completed", f"{n_done} / {len(cues)}")
            if n_done > 0:
                st.metric("Accuracy", f"{n_correct / n_done * 100:.0f}%",
                          delta=f"{n_correct} correct")

    # -- No session started yet --------------------------------------------------
    if not active:
        st.info(
            "Press **▶ Start Session** in the sidebar to begin a new "
            "closed-loop BCI session.",
            icon="🎮",
        )
        return

    cues = st.session_state[_CL_CUES]
    trial_idx = st.session_state[_CL_TRIAL_IDX]
    history = st.session_state[_CL_HISTORY]
    session_complete = trial_idx >= len(cues)

    trial_col, cursor_col = st.columns([1, 2], gap="large")

    # -- Current trial -----------------------------------------------------------
    with trial_col:
        if session_complete:
            n_correct = sum(t["correct"] for t in history)
            total = len(history)
            acc = n_correct / total * 100 if total > 0 else 0.0
            section_header("Session Complete")
            st.success(
                f"**{n_correct} / {total}** trials correct — "
                f"session accuracy **{acc:.0f}%**",
                icon="🏁",
            )
            st.caption("Press **↺ Reset Session** in the sidebar to start again.")
        else:
            cue = cues[trial_idx]
            cue_icon = _CUE_ICONS[cue]
            cue_color = _CUE_COLORS[cue]

            st.markdown(
                f"<div style='font-size:13px;opacity:0.5;margin-bottom:4px;'>"
                f"Trial {trial_idx + 1} of {len(cues)}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div style='font-size:13px;font-weight:600;margin-bottom:6px;'>"
                f"Cue</div>"
                f"<div style='font-size:64px;font-weight:900;color:{cue_color};"
                f"line-height:1;margin-bottom:2px;'>{cue_icon}</div>"
                f"<div style='font-size:20px;font-weight:700;color:{cue_color};"
                f"text-transform:uppercase;letter-spacing:.08em;"
                f"margin-bottom:16px;'>{cue}</div>",
                unsafe_allow_html=True,
            )
            st.caption("Press **▶ Next Trial** in the sidebar to decode this cue.")

        # Last completed trial result
        if history:
            last = history[-1]
            st.divider()
            st.markdown("**Last trial result**")

            pred_icon = _CUE_ICONS[last["prediction"]]
            pred_color = _CUE_COLORS[last["prediction"]]

            res_c1, res_c2 = st.columns(2)
            res_c1.markdown(
                f"<div style='font-size:11px;opacity:0.5;'>Predicted</div>"
                f"<div style='font-size:28px;font-weight:700;color:{pred_color};'>"
                f"{pred_icon} {last['prediction'].upper()}</div>",
                unsafe_allow_html=True,
            )
            res_c2.metric("Confidence", f"{last['confidence'] * 100:.0f}%")

            if last["correct"]:
                st.success(f"✓ Correct — trial {last['trial']}", icon="✅")
            else:
                st.error(f"✗ Incorrect — trial {last['trial']}", icon="❌")

            st.progress(last["confidence"],
                        text=f"Confidence: {last['confidence'] * 100:.0f}%")

    # -- Cursor trajectory -------------------------------------------------------
    with cursor_col:
        section_header("Cursor Trajectory")
        traj = st.session_state.get(_CL_TRAJ, [(0.0, 0.0)])
        render_cursor_chart(traj, title="", axis_range=12.0)

        cursor = st.session_state.get(_CL_CURSOR)
        if cursor:
            ca, cb, cc = st.columns(3)
            ca.metric("X", f"{cursor.x:.1f}")
            cb.metric("Y", f"{cursor.y:.1f}")
            cc.metric("Steps", len(traj) - 1)

    # -- Trial history table -----------------------------------------------------
    if history:
        st.divider()
        section_header("Trial History")

        n_correct = sum(t["correct"] for t in history)
        n_done = len(history)
        acc = n_correct / n_done * 100

        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Trials completed", n_done)
        sc2.metric("Correct", n_correct)
        sc3.metric("Session Accuracy", f"{acc:.0f}%")

        df = pd.DataFrame(history)
        df["cue"] = df["cue"].str.upper()
        df["prediction"] = df["prediction"].str.upper()
        df["confidence"] = df["confidence"].apply(lambda v: f"{v * 100:.0f}%")
        df["correct"] = df["correct"].apply(lambda v: "✓" if v else "✗")
        df.columns = ["Trial", "Cue", "Prediction", "Confidence", "Result"]
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Export — use raw history (not the display-formatted df) for CSV
        _export_df = pd.DataFrame(history)
        download_buttons(
            "Closed-Loop BCI Session",
            "cl_session",
            df=_export_df,
            record={
                "session_summary": {
                    "total_trials": n_done,
                    "correct": n_correct,
                    "incorrect": n_done - n_correct,
                    "accuracy_pct": round(acc, 1),
                },
                "trials": history,
            },
            extra_meta={
                "decoder_accuracy_setting": st.session_state.get("cl_accuracy", "—"),
                "n_trials_setting": st.session_state.get("cl_n_trials", "—"),
                "seed": st.session_state.get("cl_seed", "—"),
            },
        )


render()
