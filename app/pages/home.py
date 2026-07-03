"""NeuroLab — Home page."""

import streamlit as st


# ---------------------------------------------------------------------------
# CSS — one injection, class-based, no inline noise
# ---------------------------------------------------------------------------

_CSS = """
<style>
/* ---- Hero ---- */
.nl-hero {
    background: #0f0f1a;
    border-radius: 14px;
    padding: 56px 52px 48px;
    position: relative;
    overflow: hidden;
}
.nl-hero::before {
    content: "";
    position: absolute;
    top: -60px; left: 50%;
    transform: translateX(-50%);
    width: 560px; height: 360px;
    background: radial-gradient(ellipse, rgba(99,102,241,0.20) 0%, transparent 68%);
    pointer-events: none;
}
.nl-hero-eyebrow {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: .14em;
    text-transform: uppercase;
    color: #6366f1;
    margin-bottom: 16px;
}
.nl-hero-title {
    font-size: 52px;
    font-weight: 900;
    letter-spacing: -2px;
    line-height: 1.06;
    color: #f0f0ff;
    margin-bottom: 18px;
}
.nl-hero-sub {
    font-size: 18px;
    color: rgba(220,220,255,0.50);
    line-height: 1.65;
    max-width: 520px;
    margin-bottom: 28px;
}
.nl-badge {
    display: inline-block;
    background: rgba(99,102,241,0.14);
    color: #a5b4fc;
    border: 1px solid rgba(99,102,241,0.22);
    border-radius: 20px;
    padding: 3px 13px;
    font-size: 12px;
    font-weight: 500;
    margin: 3px 5px 3px 0;
}

/* ---- Shared section spacing ---- */
.nl-section {
    padding: 56px 0 16px;
}
.nl-section-alt {
    background: #F8F9FC;
    border-radius: 12px;
    padding: 48px 40px 40px;
    margin: 8px 0;
}
.nl-eyebrow {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: .14em;
    text-transform: uppercase;
    color: #6366f1;
    margin-bottom: 12px;
}
.nl-section-heading {
    font-size: 30px;
    font-weight: 700;
    letter-spacing: -.5px;
    line-height: 1.2;
    margin-bottom: 10px;
}
.nl-section-sub {
    font-size: 16px;
    opacity: 0.50;
    line-height: 1.65;
    max-width: 540px;
    margin-bottom: 40px;
}

/* ---- Feature grid (open, no box) ---- */
.nl-feature-icon {
    font-size: 26px;
    margin-bottom: 12px;
}
.nl-feature-title {
    font-size: 15px;
    font-weight: 700;
    margin-bottom: 7px;
}
.nl-feature-body {
    font-size: 14px;
    opacity: 0.55;
    line-height: 1.65;
}

/* ---- Workflow nodes (hairline only, no fill) ---- */
.nl-wf-wrap {
    display: flex;
    align-items: center;
    justify-content: center;
    flex-wrap: wrap;
    gap: 0;
    margin-top: 8px;
}
.nl-wf-node {
    border: 1px solid rgba(0,0,0,0.09);
    border-radius: 8px;
    padding: 12px 18px;
    min-width: 108px;
    text-align: center;
}
.nl-wf-node-icon { font-size: 20px; margin-bottom: 5px; }
.nl-wf-node-label {
    font-size: 11.5px;
    font-weight: 600;
    opacity: 0.65;
    letter-spacing: .02em;
}
.nl-wf-arrow {
    font-size: 14px;
    opacity: 0.25;
    padding: 0 8px;
    flex-shrink: 0;
}

/* ---- Module rows (open, no card bg) ---- */
.nl-mod-icon {
    font-size: 26px;
    margin-bottom: 10px;
    display: block;
}
.nl-mod-title {
    font-size: 16px;
    font-weight: 700;
    margin-bottom: 6px;
}
.nl-mod-body {
    font-size: 14px;
    opacity: 0.52;
    line-height: 1.65;
    margin-bottom: 10px;
}

/* ---- Thin rule between modules ---- */
.nl-rule {
    border: none;
    border-top: 1px solid rgba(0,0,0,0.07);
    margin: 28px 0;
}

/* ---- Footer band ---- */
.nl-footer {
    background: #F7F8FA;
    border-radius: 10px;
    padding: 22px 28px;
    text-align: center;
    font-size: 12px;
    opacity: 0.40;
    letter-spacing: .03em;
    margin-top: 16px;
}
</style>
"""


# ---------------------------------------------------------------------------
# HTML builders
# ---------------------------------------------------------------------------

def _hero() -> str:
    badges = "".join(
        f'<span class="nl-badge">{t}</span>'
        for t in ["Neural Simulation", "EEG Analysis", "BCI Decoding", "Deep Learning"]
    )
    return f"""
    <div class="nl-hero">
        <div class="nl-hero-eyebrow">Open-source · Research-grade · Extensible</div>
        <div class="nl-hero-title">Build the future<br>of brain interfaces.</div>
        <div class="nl-hero-sub">
            NeuroLab is an interactive research platform for computational neuroscience
            and brain-computer interface development — from single-neuron simulation
            to closed-loop EEG decoding.
        </div>
        {badges}
    </div>
    """


def _feature(icon: str, title: str, body: str) -> str:
    return (
        f'<div class="nl-feature-icon">{icon}</div>'
        f'<div class="nl-feature-title">{title}</div>'
        f'<div class="nl-feature-body">{body}</div>'
    )


def _workflow() -> str:
    steps = [
        ("📡", "Neural Data"),
        ("⚙️", "Preprocess"),
        ("🤖", "EEGNet"),
        ("🎯", "Prediction"),
        ("🎮", "Cursor"),
    ]
    nodes = '<span class="nl-wf-arrow">→</span>'.join(
        f'<div class="nl-wf-node">'
        f'<div class="nl-wf-node-icon">{icon}</div>'
        f'<div class="nl-wf-node-label">{label}</div>'
        f'</div>'
        for icon, label in steps
    )
    return f'<div class="nl-wf-wrap">{nodes}</div>'


def _module(icon: str, title: str, body: str) -> str:
    return (
        f'<span class="nl-mod-icon">{icon}</span>'
        f'<div class="nl-mod-title">{title}</div>'
        f'<div class="nl-mod-body">{body}</div>'
    )


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

def render() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)

    # ── Hero ────────────────────────────────────────────────────────────────
    st.markdown(_hero(), unsafe_allow_html=True)

    cta1, cta2, _ = st.columns([1, 1, 4])
    with cta1:
        st.page_link("pages/neurosim.py",  label="▶  Start Simulating")
    with cta2:
        st.page_link("pages/eeg_studio.py", label="📡  Explore EEG Data")

    # ── What it is — open feature grid ──────────────────────────────────────
    st.markdown('<div class="nl-section">', unsafe_allow_html=True)
    st.markdown(
        '<div class="nl-eyebrow">Platform</div>'
        '<div class="nl-section-heading">Everything in one place.</div>'
        '<div class="nl-section-sub">From a single neuron&#39;s membrane potential '
        'to a closed-loop BCI decoding real EEG signals — NeuroLab covers '
        'the full stack, interactively.</div>',
        unsafe_allow_html=True,
    )

    f1, f2, f3 = st.columns(3, gap="large")
    with f1:
        st.markdown(_feature(
            "🔬", "Live Neural Simulation",
            "Tune membrane resistance, threshold, and time constant. "
            "Watch spike trains form or disappear in real time.",
        ), unsafe_allow_html=True)
    with f2:
        st.markdown(_feature(
            "📊", "Real EEG Data",
            "4,897 trials from a 64-channel motor imagery dataset. "
            "Inspect raw signals before any processing.",
        ), unsafe_allow_html=True)
    with f3:
        st.markdown(_feature(
            "🤖", "Deep Learning Decoder",
            "EEGNet — a compact depthwise CNN — runs inference in the browser. "
            "Calibrated softmax probabilities, no black box.",
        ), unsafe_allow_html=True)

    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)

    f4, f5, f6 = st.columns(3, gap="large")
    with f4:
        st.markdown(_feature(
            "🎮", "Closed-Loop Control",
            "Decoded intent drives a virtual cursor. The control loop "
            "is tangible, inspectable, and real.",
        ), unsafe_allow_html=True)
    with f5:
        st.markdown(_feature(
            "🧩", "Extensible Package",
            "Abstract base classes for neurons, output devices, and datasets. "
            "Add your own model in minutes.",
        ), unsafe_allow_html=True)
    with f6:
        st.markdown(_feature(
            "📚", "Structured Tutorials",
            "Learning tracks from biophysical modelling to BCI pipeline design, "
            "written for researchers and students.",
        ), unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ── BCI Workflow — subtle alt background, hairline nodes ─────────────────
    st.markdown('<div class="nl-section-alt">', unsafe_allow_html=True)
    st.markdown(
        '<div class="nl-eyebrow">Pipeline</div>'
        '<div class="nl-section-heading">How a BCI works.</div>'
        '<div class="nl-section-sub">NeuroLab makes each stage of the decode-and-control '
        'pipeline interactive and visible.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(_workflow(), unsafe_allow_html=True)

    st.markdown("<div style='height:36px'></div>", unsafe_allow_html=True)

    w1, w2, w3 = st.columns(3, gap="large")
    with w1:
        st.markdown(
            "**Acquire & inspect**  \n"
            "EEG Studio shows the raw 64-channel signal for any trial. "
            "Understanding what you're decoding starts here."
        )
    with w2:
        st.markdown(
            "**Decode**  \n"
            "EEGNet classifies each trial into a movement class. "
            "Decoder Lab shows you the probabilities and where the model is uncertain."
        )
    with w3:
        st.markdown(
            "**Control**  \n"
            "The prediction moves a cursor. Closed-Loop BCI makes the output "
            "physical and immediate — the loop is closed."
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Modules — open list, no card backgrounds ─────────────────────────────
    st.markdown('<div class="nl-section">', unsafe_allow_html=True)
    st.markdown(
        '<div class="nl-eyebrow">Modules</div>'
        '<div class="nl-section-heading">Start exploring.</div>',
        unsafe_allow_html=True,
    )

    m1, m2, m3 = st.columns(3, gap="large")
    with m1:
        st.markdown(_module(
            "🔬", "NeuroSim",
            "Interactive LIF neuron simulator. Adjust parameters and watch "
            "the voltage trace and spike raster update instantly.",
        ), unsafe_allow_html=True)
        st.page_link("pages/neurosim.py", label="Open NeuroSim →")

    with m2:
        st.markdown(_module(
            "📡", "EEG Studio",
            "Browse motor imagery EEG trials from real participants. "
            "Visualise raw multichannel signals before decoding.",
        ), unsafe_allow_html=True)
        st.page_link("pages/eeg_studio.py", label="Open EEG Studio →")

    with m3:
        st.markdown(_module(
            "🤖", "Decoder Lab",
            "Run EEGNet on any trial. Compare predicted class against ground "
            "truth and move a cursor with each inference.",
        ), unsafe_allow_html=True)
        st.page_link("pages/decoder_lab.py", label="Open Decoder Lab →")

    st.markdown('<hr class="nl-rule">', unsafe_allow_html=True)

    m4, m5, m6 = st.columns(3, gap="large")
    with m4:
        st.markdown(_module(
            "🎮", "Closed-Loop BCI",
            "A mock decoder generates movement commands in real time. "
            "See the full pipeline — signal to cursor — in one place.",
        ), unsafe_allow_html=True)
        st.page_link("pages/closed_loop.py", label="Open Closed-Loop →")

    with m5:
        st.markdown(_module(
            "📚", "Tutorials",
            "Five learning tracks from neuron modelling to BCI design. "
            "Structured for students and researchers.",
        ), unsafe_allow_html=True)
        st.page_link("pages/tutorials.py", label="Open Tutorials →")

    with m6:
        st.markdown(_module(
            "📦", "NeuroSim Package",
            "The open-source Python library behind NeuroLab. "
            "Extend it with custom neurons, decoders, and output devices.",
        ), unsafe_allow_html=True)
        st.link_button("View source →", "https://github.com/")

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Footer ───────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="nl-footer">'
        'NeuroLab &nbsp;·&nbsp; Built with Streamlit &nbsp;·&nbsp; '
        'Powered by NeuroSim &nbsp;·&nbsp; Open Source'
        '</div>',
        unsafe_allow_html=True,
    )

