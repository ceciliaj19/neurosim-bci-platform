"""NeuroLab — Scientific References page."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from components.page_header import render_page_header
from components.ui import section_header

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Ref:
    """A single bibliographic reference."""

    title: str
    authors: str
    year: int
    description: str
    category: str
    doi: str | None = None   # bare DOI, e.g. "10.1109/TNN.2003.820440"
    url: str | None = None   # fallback for open-access books / websites
    used_in: tuple[str, ...] = field(default_factory=tuple)

    @property
    def link(self) -> str | None:
        if self.doi:
            return f"https://doi.org/{self.doi}"
        return self.url

    def matches(self, query: str) -> bool:
        """Return True if *query* appears (case-insensitively) in searchable text."""
        haystack = f"{self.title} {self.authors} {self.description}".lower()
        return query.lower() in haystack


# ---------------------------------------------------------------------------
# Category ordering
# ---------------------------------------------------------------------------

CATEGORIES: list[str] = [
    "Computational Neuroscience",
    "Neuron Models",
    "Brain-Computer Interfaces",
    "EEG Signal Processing",
    "Deep Learning for EEG",
    "Datasets",
]

# ---------------------------------------------------------------------------
# Reference list
# ---------------------------------------------------------------------------

REFS: list[Ref] = [
    # ── Computational Neuroscience ───────────────────────────────────────────
    Ref(
        title="Theoretical Neuroscience: Computational and Mathematical Modeling "
              "of Neural Systems",
        authors="Dayan, P. & Abbott, L. F.",
        year=2001,
        description=(
            "The standard graduate-level textbook for computational neuroscience. "
            "Covers neural encoding and decoding, information theory, network "
            "models, and learning rules. Foundational reading for understanding "
            "the single-neuron models implemented in NeuroSim."
        ),
        category="Computational Neuroscience",
        url="https://mitpress.mit.edu/9780262541855/theoretical-neuroscience/",
        used_in=("NeuroSim",),
    ),
    Ref(
        title="Neuronal Dynamics: From Single Neurons to Networks and Models "
              "of Cognition",
        authors="Gerstner, W., Kistler, W. M., Naud, R. & Paninski, L.",
        year=2014,
        description=(
            "A comprehensive and freely available textbook covering spiking "
            "neuron models (LIF, adaptive, Hodgkin-Huxley), synaptic dynamics, "
            "and network-level phenomena. Available in full online at "
            "neuronaldynamics.epfl.ch."
        ),
        category="Computational Neuroscience",
        url="https://neuronaldynamics.epfl.ch/",
        used_in=("NeuroSim",),
    ),
    Ref(
        title="Methods in Neuronal Modeling: From Ions to Networks",
        authors="Koch, C. & Segev, I. (eds.)",
        year=1998,
        description=(
            "Edited volume covering biophysical models of single neurons and "
            "synapses, compartmental modelling, and analysis of neural circuits. "
            "Reference for the conductance-based modelling approach that "
            "underpins the Hodgkin-Huxley and Izhikevich models."
        ),
        category="Computational Neuroscience",
        url="https://mitpress.mit.edu/9780262112314/methods-in-neuronal-modeling/",
    ),

    # ── Neuron Models ────────────────────────────────────────────────────────
    Ref(
        title="Recherches quantitatives sur l'excitation électrique des nerfs "
              "traitée comme une polarisation",
        authors="Lapicque, L.",
        year=1907,
        description=(
            "The original publication introducing the integrate-and-fire (IF) "
            "neuron model — a capacitor charged by injected current that fires "
            "when it reaches a threshold voltage. The direct predecessor of the "
            "LIF model implemented in NeuroSim."
        ),
        category="Neuron Models",
        url="https://fr.wikipedia.org/wiki/Louis_Lapicque",
        used_in=("NeuroSim",),
    ),
    Ref(
        title="A Quantitative Description of Membrane Current and Its Application "
              "to Conduction and Excitation in Nerve",
        authors="Hodgkin, A. L. & Huxley, A. F.",
        year=1952,
        description=(
            "Nobel Prize-winning paper establishing the conductance-based model "
            "of the action potential. Introduced voltage-gated Na⁺ and K⁺ "
            "channels modelled by nonlinear differential equations, forming the "
            "biophysical foundation of all modern spiking neuron models."
        ),
        category="Neuron Models",
        doi="10.1113/jphysiol.1952.sp004764",
        used_in=("NeuroSim",),
    ),
    Ref(
        title="A Review of the Integrate-and-Fire Neuron Model: "
              "I. Homogeneous Synaptic Input",
        authors="Burkitt, A. N.",
        year=2006,
        description=(
            "Systematic review of LIF model variants, their analytical solutions, "
            "and firing-rate approximations. Useful reference for understanding "
            "the theoretical properties of the LIF neuron implemented in NeuroSim."
        ),
        category="Neuron Models",
        doi="10.1007/s00422-006-0068-6",
        used_in=("NeuroSim",),
    ),
    Ref(
        title="Simple Model of Spiking Neurons",
        authors="Izhikevich, E. M.",
        year=2003,
        description=(
            "Introduces a two-variable model combining the biological plausibility "
            "of Hodgkin-Huxley with the computational efficiency of integrate-and-fire. "
            "Table 1 of this paper is the direct source of the six Izhikevich presets "
            "(Regular Spiking, Intrinsically Bursting, Chattering, Fast Spiking, "
            "Low-Threshold Spiking, Resonator) implemented in NeuroSim."
        ),
        category="Neuron Models",
        doi="10.1109/TNN.2003.820440",
        used_in=("NeuroSim",),
    ),

    # ── Brain-Computer Interfaces ────────────────────────────────────────────
    Ref(
        title="Brain–Computer Interfaces for Communication and Control",
        authors="Wolpaw, J. R., Birbaumer, N., McFarland, D. J., Pfurtscheller, G. "
                "& Vaughan, T. M.",
        year=2002,
        description=(
            "Landmark review paper defining the BCI field. Covers EEG-based "
            "paradigms including P300, SSVEP, and motor imagery; signal processing "
            "pipelines; and clinical applications. The canonical reference for "
            "the closed-loop BCI architecture demonstrated in NeuroLab."
        ),
        category="Brain-Computer Interfaces",
        doi="10.1016/S1388-2457(02)00057-3",
        used_in=("Closed-Loop BCI",),
    ),
    Ref(
        title="Motor Imagery and Direct Brain-Computer Communication",
        authors="Pfurtscheller, G. & Neuper, C.",
        year=2001,
        description=(
            "Establishes the motor imagery paradigm used in NeuroLab's dataset: "
            "imagining left- or right-hand movement produces lateralised "
            "suppression of alpha/beta power (ERD) over the motor cortex, "
            "which the EEGNet decoder learns to distinguish."
        ),
        category="Brain-Computer Interfaces",
        doi="10.1109/5.939829",
        used_in=("Closed-Loop BCI", "EEG Studio"),
    ),
    Ref(
        title="The BCI Competition III: Validating Alternative Approaches "
              "to Actual BCI Problems",
        authors="Blankertz, B., Müller, K.-R., Krusienski, D. J., Schalk, G., "
                "Wolpaw, J. R., Schlögl, A., Pfurtscheller, G., Millán, J. R., "
                "Schröder, M. & Birbaumer, N.",
        year=2006,
        description=(
            "Describes the BCI Competition benchmark datasets and evaluation "
            "framework widely used in motor imagery classification research. "
            "Establishes evaluation conventions (accuracy, kappa) adopted in "
            "NeuroLab's Decoder Lab."
        ),
        category="Brain-Computer Interfaces",
        doi="10.1109/TNSRE.2006.875642",
        used_in=("Decoder Lab",),
    ),
    Ref(
        title="Brain-Computer Interface Research: A State-of-the-Art Summary 11",
        authors="Müller-Putz, G., Krausz, G., Gantner, I., Stippich, H. & Sellers, E.",
        year=2023,
        description=(
            "Annual state-of-the-art summary from the BCI Society covering recent "
            "advances in non-invasive EEG-based BCIs, hybrid systems, and "
            "clinical translation. Useful for contextualising where NeuroLab's "
            "motor imagery demo sits within the broader BCI research landscape."
        ),
        category="Brain-Computer Interfaces",
        doi="10.1007/978-3-031-49457-4",
    ),

    # ── EEG Signal Processing ────────────────────────────────────────────────
    Ref(
        title="Event-Related EEG/MEG Synchronization and Desynchronization: "
              "Basic Principles",
        authors="Pfurtscheller, G. & Lopes da Silva, F. H.",
        year=1999,
        description=(
            "Defines event-related desynchronisation (ERD) and synchronisation "
            "(ERS) — the alpha/beta power changes visible in the EEG Studio "
            "spectrogram and band-power plots during motor imagery. Essential "
            "background for interpreting EEG features used in BCI decoding."
        ),
        category="EEG Signal Processing",
        doi="10.1016/S1388-2457(99)00141-8",
        used_in=("EEG Studio",),
    ),
    Ref(
        title="Optimizing Spatial Filters for Robust EEG Single-Trial Analysis",
        authors="Blankertz, B., Tomioka, R., Lemm, S., Kawanabe, M. & Müller, K.-R.",
        year=2008,
        description=(
            "Introduces Common Spatial Patterns (CSP), the classical spatial "
            "filtering algorithm for motor imagery classification. CSP maximises "
            "the variance ratio between two classes, producing features "
            "complementary to EEGNet's learned spatial filters."
        ),
        category="EEG Signal Processing",
        doi="10.1109/MSP.2008.4408441",
        used_in=("EEG Studio", "Decoder Lab"),
    ),
    Ref(
        title="Independent Component Analysis of Electroencephalographic Data",
        authors="Makeig, S., Bell, A. J., Jung, T.-P. & Sejnowski, T. J.",
        year=1996,
        description=(
            "Seminal application of ICA to EEG, demonstrating that eye-blink "
            "and muscle artefacts can be isolated and removed as independent "
            "components. The standard reference for EEG artefact rejection "
            "preprocessing pipelines."
        ),
        category="EEG Signal Processing",
        url="https://proceedings.neurips.cc/paper/1996/hash/"
            "e7a425c6ece20cbc9056c98cbe5b18ca-Abstract.html",
    ),

    # ── Deep Learning for EEG ────────────────────────────────────────────────
    Ref(
        title="EEGNet: A Compact Convolutional Neural Network for EEG-Based "
              "Brain-Computer Interfaces",
        authors="Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M., "
                "Hung, C. P. & Lance, B. J.",
        year=2018,
        description=(
            "Introduces EEGNet — a depthwise-separable CNN architecture for "
            "EEG classification that achieves competitive accuracy with very few "
            "parameters (~2,500 trainable weights for the 2-class case). "
            "This is the exact architecture implemented in NeuroLab's "
            "`neurosim/networks/eegnet.py` and trained via `scripts/train_eegnet.py`."
        ),
        category="Deep Learning for EEG",
        doi="10.1088/1741-2552/aace8c",
        used_in=("Decoder Lab",),
    ),
    Ref(
        title="Deep Learning With Convolutional Neural Networks for EEG Decoding "
              "and Visualization",
        authors="Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J., "
                "Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F., "
                "Burgard, W. & Ball, T.",
        year=2017,
        description=(
            "Systematic comparison of deep ConvNet architectures for EEG decoding, "
            "including ShallowConvNet and DeepConvNet. Establishes that CNNs can "
            "match or exceed classical methods on motor imagery and introduces "
            "feature visualisation techniques for EEG models."
        ),
        category="Deep Learning for EEG",
        doi="10.1002/hbm.23730",
        used_in=("Decoder Lab",),
    ),
    Ref(
        title="Deep Learning-Based Electroencephalography Analysis: A Systematic Review",
        authors="Roy, Y., Banville, H., Albuquerque, I., Gramfort, A., "
                "Falk, T. H. & Faubert, J.",
        year=2019,
        description=(
            "Comprehensive review of deep learning methods applied to EEG, "
            "covering CNN, RNN, and autoencoder architectures across BCI, "
            "seizure detection, and sleep staging applications. Useful for "
            "contextualising EEGNet within the broader landscape of EEG deep learning."
        ),
        category="Deep Learning for EEG",
        doi="10.1088/1741-2552/ab260c",
    ),

    # ── Datasets ─────────────────────────────────────────────────────────────
    Ref(
        title="PhysioBank, PhysioToolkit, and PhysioNet: Components of a New "
              "Research Resource for Complex Physiologic Signals",
        authors="Goldberger, A. L., Amaral, L. A. N., Glass, L., Hausdorff, J. M., "
                "Ivanov, P. C., Mark, R. G., Mietus, J. E., Moody, G. B., "
                "Peng, C.-K. & Stanley, H. E.",
        year=2000,
        description=(
            "Describes the PhysioNet resource — the public repository hosting the "
            "EEG Motor Movement/Imagery Dataset (EEGMMI) used in NeuroLab. "
            "PhysioNet provides open-access physiological datasets and analysis "
            "software for biomedical research."
        ),
        category="Datasets",
        doi="10.1161/01.CIR.101.23.e215",
        url="https://physionet.org/",
        used_in=("EEG Studio", "Decoder Lab"),
    ),
    Ref(
        title="BCI2000: A General-Purpose Brain-Computer Interface (BCI) System",
        authors="Schalk, G., McFarland, D. J., Hinterberger, T., Birbaumer, N. "
                "& Wolpaw, J. R.",
        year=2004,
        description=(
            "Describes BCI2000, the software system used to record the PhysioNet "
            "EEG Motor Movement/Imagery Dataset (64-channel, 160 Hz, 109 subjects) "
            "that NeuroLab uses for EEG analysis and EEGNet training. "
            "All EEGMMI trials were acquired and epoch-labelled using BCI2000."
        ),
        category="Datasets",
        doi="10.1109/TBME.2004.827072",
        used_in=("EEG Studio", "Decoder Lab"),
    ),
]

# ---------------------------------------------------------------------------
# Helper renderers
# ---------------------------------------------------------------------------

_MODULE_COLORS: dict[str, str] = {
    "NeuroSim":        "#6366f1",
    "EEG Studio":      "#10b981",
    "Decoder Lab":     "#f59e0b",
    "Closed-Loop BCI": "#c5221f",
    "Dashboard":       "#1967d2",
}


def _module_chip(name: str) -> str:
    color = _MODULE_COLORS.get(name, "#888")
    return (
        f"<span style='"
        f"display:inline-block;"
        f"background:rgba(0,0,0,0.05);"
        f"border:1px solid {color}44;"
        f"color:{color};"
        f"border-radius:4px;"
        f"padding:1px 8px;"
        f"font-size:11px;"
        f"font-weight:600;"
        f"margin-right:4px;"
        f"'>{name}</span>"
    )


def _render_ref(ref: Ref) -> None:
    """Render one reference entry."""
    title_col, link_col = st.columns([6, 1])

    with title_col:
        st.markdown(f"**{ref.title}**")
        st.caption(f"{ref.authors} · {ref.year}")

    with link_col:
        if ref.link:
            st.link_button("↗ Open", ref.link, use_container_width=True)

    st.markdown(ref.description)

    footer_parts: list[str] = []

    if ref.doi:
        footer_parts.append(f"DOI: `{ref.doi}`")

    if ref.used_in:
        chips = "".join(_module_chip(m) for m in ref.used_in)
        footer_parts.append(f"<span style='font-size:11px;opacity:0.55;'>Used in:</span> {chips}")

    if footer_parts:
        st.markdown(
            "  ·  ".join(footer_parts),
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

def render() -> None:
    render_page_header(
        "📖", "Scientific References",
        "Papers, textbooks, and datasets underlying NeuroLab",
    )

    # -- Sidebar: category filter --------------------------------------------
    with st.sidebar:
        st.markdown("**Filter by Category**")

        sb_col1, sb_col2 = st.columns(2)
        select_all = sb_col1.button("Select all", use_container_width=True, key="ref_sel_all")
        clear_all  = sb_col2.button("Clear all",  use_container_width=True, key="ref_clr_all")

        if "ref_selected_cats" not in st.session_state or select_all:
            st.session_state["ref_selected_cats"] = list(CATEGORIES)
        if clear_all:
            st.session_state["ref_selected_cats"] = []

        selected_cats: list[str] = st.multiselect(
            "Categories",
            options=CATEGORIES,
            default=st.session_state.get("ref_selected_cats", CATEGORIES),
            label_visibility="collapsed",
            key="ref_cat_multi",
        )
        # Keep session state in sync with the multiselect
        st.session_state["ref_selected_cats"] = selected_cats

        st.divider()
        st.markdown("**Summary**")
        st.metric("Total references", len(REFS))
        st.metric("Categories", len(CATEGORIES))

    # -- Search bar ----------------------------------------------------------
    query = st.text_input(
        "Search",
        placeholder="Search by title, author, or keyword…",
        label_visibility="collapsed",
        key="ref_query",
    )

    # -- Filter --------------------------------------------------------------
    filtered = [
        r for r in REFS
        if r.category in selected_cats and (not query or r.matches(query))
    ]

    cats_shown = [c for c in CATEGORIES if any(r.category == c for r in filtered)]

    st.caption(
        f"Showing **{len(filtered)}** of {len(REFS)} references "
        f"across {len(cats_shown)} categor{'y' if len(cats_shown) == 1 else 'ies'}."
    )

    if not filtered:
        st.info(
            "No references match the current search and filter. "
            "Try a different keyword or select more categories.",
            icon="🔍",
        )
        return

    # -- Render by category --------------------------------------------------
    for cat in CATEGORIES:
        cat_refs = [r for r in filtered if r.category == cat]
        if not cat_refs:
            continue

        st.divider()
        section_header(cat)

        for i, ref in enumerate(cat_refs):
            _render_ref(ref)
            if i < len(cat_refs) - 1:
                st.markdown(
                    "<hr style='border:none;border-top:1px solid rgba(0,0,0,0.07);"
                    "margin:12px 0;'>",
                    unsafe_allow_html=True,
                )

    # -- Footer --------------------------------------------------------------
    st.divider()
    st.markdown(
        "<div style='text-align:center;font-size:12px;opacity:0.35;'>"
        "DOI links open via doi.org · "
        "Open-access links open the publisher or repository page · "
        "All external links open in a new tab"
        "</div>",
        unsafe_allow_html=True,
    )


render()
