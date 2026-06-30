# NeuroSim Design Decisions

This document records the major decisions made during the development of NeuroSim.

The purpose is to make the project easier to understand, maintain, and extend.

---

# Decision 1 — Modular Python Package

**Decision**

NeuroSim is implemented as an installable Python package rather than a collection of notebooks or scripts.

**Why**

* Reusable by other projects
* Easy installation using `pip install -e .`
* Clear separation between library code and applications

---

# Decision 2 — Interactive Web App

**Decision**

The user interface is built separately from the simulation engine using Streamlit.

**Why**

The computational models should remain independent from the visualization layer.

This allows:

* command-line usage
* notebooks
* future desktop applications
* future web applications

without changing the simulation code.

---

# Decision 3 — SimulationResult Object

**Decision**

All simulations return a `SimulationResult` object instead of a dictionary.

**Why**

This provides a consistent API for:

* plotting
* statistics
* exporting
* analysis

Future neuron models will return the same object.

---

# Future Decisions

This section will be updated as NeuroSim evolves.
