# NeuroSim Roadmap

NeuroSim is an interactive computational neuroscience and BCI learning platform built as a modular Python package and web application.

## v0.1 — Core LIF Neuron Demo

**Goal:** Build the first complete, demoable version of NeuroSim.

* Implement Leaky Integrate-and-Fire neuron model
* Create installable Python package structure
* Add example script for LIF simulation
* Build Streamlit app with interactive sliders
* Display membrane voltage, spike count, and firing rate
* Add basic README documentation and screenshots

## v0.2 — Analysis Tools

**Goal:** Add reusable analysis functions for spike trains.

* Compute inter-spike intervals
* Generate firing-rate curves
* Add current-vs-firing-rate plots
* Add reusable visualization utilities
* Add unit tests for LIF model behavior

## v0.3 — Additional Neuron Models

**Goal:** Compare multiple computational neuron models.

* Implement Izhikevich neuron model
* Implement Hodgkin-Huxley neuron model
* Add model selector to Streamlit app
* Compare voltage dynamics across models
* Add educational explanations for each model

## v0.4 — Network Simulation

**Goal:** Move from single neurons to small neural circuits.

* Implement simple excitatory/inhibitory networks
* Add synaptic connections
* Generate spike raster plots
* Visualize population firing rate
* Add network parameter controls

## v0.5 — BCI Integration

**Goal:** Connect computational neuroscience simulations to real neural data.

* Add EEG preprocessing demo
* Integrate motor imagery classification pipeline
* Visualize EEG decoding outputs
* Compare simulated neural activity with real EEG-based BCI concepts

## v1.0 — Public Release

**Goal:** Release NeuroSim as a polished open-source educational neuroscience tool.

* Complete documentation
* Add tutorial notebooks
* Add polished README with screenshots/GIFs
* Add test coverage
* Deploy Streamlit app
* Publish project demo video