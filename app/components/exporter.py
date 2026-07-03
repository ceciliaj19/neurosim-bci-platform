"""NeuroLab — reusable experiment export utility.

Public API
----------
download_buttons(label, filename_stem, df=None, record=None)
    Renders an expander containing Streamlit download buttons for CSV and/or
    JSON. Call it immediately after the chart or table whose data you want to
    export.

Usage
-----
::

    from components.exporter import download_buttons

    # Tabular data — produces CSV + JSON buttons
    download_buttons("F-I Curve", "fi_curve", df=fi_df)

    # Record data — produces JSON button only
    download_buttons("Last Prediction", "prediction", record=pred_dict)

    # Both — CSV from df, richer JSON from record
    download_buttons("Session", "session", df=history_df, record=summary_dict)
"""

from __future__ import annotations

import io
import json
from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st

_PLATFORM = "NeuroLab v0.1.0"


# ---------------------------------------------------------------------------
# Internal serialisers
# ---------------------------------------------------------------------------

def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _make_csv(df: pd.DataFrame) -> bytes:
    """Serialise *df* to UTF-8 CSV bytes (no BOM, no index)."""
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def _make_json(
    data: dict | list,
    label: str,
    extra_meta: dict[str, Any] | None = None,
) -> bytes:
    """Wrap *data* in a metadata envelope and serialise to UTF-8 JSON bytes."""
    meta: dict[str, Any] = {
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "platform": _PLATFORM,
        "label": label,
    }
    if extra_meta:
        meta.update(extra_meta)

    payload = {"_meta": meta, "data": data}
    return json.dumps(payload, indent=2, default=str).encode("utf-8")


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------

def download_buttons(
    label: str,
    filename_stem: str,
    df: pd.DataFrame | None = None,
    record: dict | list | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> None:
    """Render CSV and/or JSON download buttons inside a collapsed expander.

    Parameters
    ----------
    label : str
        Human-readable name shown in the expander title and embedded in the
        JSON ``_meta.label`` field.
    filename_stem : str
        Base filename without extension or timestamp.  The generated filenames
        follow the pattern ``{stem}_{YYYYMMDD_HHMMSS}.{ext}``.
    df : pd.DataFrame, optional
        Tabular data.  When provided a **CSV** button is always shown.
        A **JSON** button is also shown (using ``df.to_dict(orient="records")``
        as the payload) unless *record* is also supplied, in which case
        *record* is used for the JSON instead.
    record : dict or list, optional
        Arbitrary JSON-serialisable data.  When provided a **JSON** button is
        shown.  Overrides the ``df``-derived JSON payload when both are given,
        allowing callers to embed richer metadata alongside the tabular data.
    extra_meta : dict, optional
        Additional key/value pairs merged into the ``_meta`` block of the JSON
        export (e.g. neuron parameters, session settings).
    """
    if df is None and record is None:
        return

    ts = _timestamp()

    with st.expander(f"⬇ Export — {label}"):
        show_csv = df is not None
        # JSON payload: prefer explicit record over df-derived list
        json_payload: dict | list | None = None
        if record is not None:
            json_payload = record
        elif df is not None:
            json_payload = df.to_dict(orient="records")

        n_buttons = int(show_csv) + int(json_payload is not None)
        cols = st.columns(n_buttons) if n_buttons > 1 else [st.container()]

        col_idx = 0

        if show_csv:
            with cols[col_idx]:
                st.download_button(
                    label="⬇ CSV",
                    data=_make_csv(df),
                    file_name=f"{filename_stem}_{ts}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key=f"_dl_csv_{filename_stem}_{ts}",
                )
            col_idx += 1

        if json_payload is not None:
            with cols[col_idx]:
                st.download_button(
                    label="⬇ JSON",
                    data=_make_json(json_payload, label, extra_meta),
                    file_name=f"{filename_stem}_{ts}.json",
                    mime="application/json",
                    use_container_width=True,
                    key=f"_dl_json_{filename_stem}_{ts}",
                )
