"""Published parameter presets for the Izhikevich neuron model.

All values are taken from:

    Izhikevich, E.M. (2003). Simple model of spiking neurons.
    *IEEE Transactions on Neural Networks*, 14(6), 1569-1572.
    https://doi.org/10.1109/TNN.2003.820440

See Table 1 and Figure 1 of that paper for the original parameter sets and
corresponding voltage traces.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IzhikevichPreset:
    """Published parameter set for the Izhikevich two-variable neuron model.

    Parameters
    ----------
    a : float
        Time scale of the recovery variable ``u``.
    b : float
        Sensitivity of ``u`` to subthreshold membrane oscillations.
    c : float
        After-spike reset value of the membrane potential ``v`` (mV).
    d : float
        After-spike increment applied to the recovery variable ``u``.
    description : str
        One-line description of the firing pattern and typical cell type.
    """

    a: float
    b: float
    c: float
    d: float
    description: str


# ---------------------------------------------------------------------------
# Registry — keys are lowercase canonical names
# ---------------------------------------------------------------------------

_PRESETS: dict[str, IzhikevichPreset] = {
    "regular spiking": IzhikevichPreset(
        a=0.02, b=0.2, c=-65.0, d=8.0,
        description=(
            "Tonic spiking with spike-frequency adaptation. "
            "Typical of excitatory cortical pyramidal neurons."
        ),
    ),
    "intrinsically bursting": IzhikevichPreset(
        a=0.02, b=0.2, c=-55.0, d=4.0,
        description=(
            "Initial high-frequency burst followed by regular spiking. "
            "Found in layer 5 cortical pyramidal neurons."
        ),
    ),
    "chattering": IzhikevichPreset(
        a=0.02, b=0.2, c=-50.0, d=2.0,
        description=(
            "Rhythmic high-frequency bursts separated by silent periods. "
            "Found in layer 2/3 and layer 4 cortical neurons."
        ),
    ),
    "fast spiking": IzhikevichPreset(
        a=0.1, b=0.2, c=-65.0, d=2.0,
        description=(
            "High-frequency tonic spiking with no adaptation. "
            "Typical of GABAergic inhibitory interneurons."
        ),
    ),
    "low threshold spiking": IzhikevichPreset(
        a=0.02, b=0.25, c=-65.0, d=2.0,
        description=(
            "Fires at low input currents; slower adaptation than RS. "
            "Associated with certain inhibitory interneuron classes."
        ),
    ),
    "resonator": IzhikevichPreset(
        a=0.1, b=0.26, c=-65.0, d=-1.0,
        description=(
            "Subthreshold oscillations; responds selectively to inputs "
            "near its resonant frequency. Found in some cortical and "
            "thalamic neurons."
        ),
    ),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_izhikevich_preset(name: str) -> IzhikevichPreset:
    """Return a published Izhikevich parameter preset by name.

    Lookup is case-insensitive and strips leading/trailing whitespace.

    Parameters
    ----------
    name : str
        Preset name.  Use :func:`list_presets` for valid choices.

    Returns
    -------
    IzhikevichPreset
        Frozen dataclass with fields ``a``, ``b``, ``c``, ``d``,
        and ``description``.

    Raises
    ------
    KeyError
        If *name* does not match any registered preset.  The error message
        lists all valid preset names.

    Examples
    --------
    >>> preset = get_izhikevich_preset("Regular Spiking")
    >>> preset.a, preset.b, preset.c, preset.d
    (0.02, 0.2, -65.0, 8.0)

    >>> get_izhikevich_preset("unknown")
    Traceback (most recent call last):
        ...
    KeyError: "Unknown preset 'unknown'. Available: ..."
    """
    key = name.strip().lower()
    if key not in _PRESETS:
        available = ", ".join(f"'{k}'" for k in _PRESETS)
        raise KeyError(f"Unknown preset {name!r}. Available: {available}.")
    return _PRESETS[key]


def list_presets() -> list[str]:
    """Return preset names in their canonical display form (title case).

    Returns
    -------
    list[str]
        Ordered list of preset names suitable for use in a UI selectbox.

    Examples
    --------
    >>> list_presets()
    ['Regular Spiking', 'Intrinsically Bursting', 'Chattering',
     'Fast Spiking', 'Low Threshold Spiking', 'Resonator']
    """
    return [name.title() for name in _PRESETS]
