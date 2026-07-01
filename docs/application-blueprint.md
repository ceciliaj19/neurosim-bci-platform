# NeuroLab Application Blueprint

NeuroLab is an interactive platform for computational neuroscience, neural signal analysis, BCI decoding, and closed-loop control demos.

## Core Modules

### 1. Home

Purpose:
Introduce NeuroLab and guide users into major workflows.

Features:
- Project overview
- Quick start buttons
- Recent simulations
- Demo workflows

---

### 2. NeuroSim

Purpose:
Explore computational neuron models.

Features:
- Leaky Integrate-and-Fire model
- Izhikevich model
- Hodgkin-Huxley model
- Voltage plots
- Spike statistics
- F-I curves
- Network simulations

---

### 3. EEG Studio

Purpose:
Inspect and preprocess neural recordings.

Features:
- Load EEG data
- View raw EEG
- Filter signals
- Inspect channels
- Visualize frequency content
- Mark artifacts

---

### 4. Decoder Lab

Purpose:
Train and evaluate BCI decoders.

Features:
- CSP + LDA
- EEGNet
- CNN baselines
- Cross-validation
- Confusion matrix
- Prediction confidence
- Model comparison

---

### 5. Closed-Loop Demo

Purpose:
Show how decoded neural intent controls an output device.

Features:
- Mock decoder first
- Cursor controller
- Prediction confidence
- Cursor trajectory
- Trial replay
- Real EEG integration later

---

### 6. Benchmark Dashboard

Purpose:
Compare BCI models and pipelines.

Features:
- Accuracy
- Training time
- Inference time
- Confusion matrix
- Dataset summary
- Model comparison table

---

## Version 0.1 App Scope

The first polished app should include:

- NeuroSim LIF demo
- Closed-loop cursor demo using mock predictions
- Clean navigation
- Explanation panels
- Basic metrics

## Long-Term Vision

NeuroLab should connect:

```text
Neural data
→ preprocessing
→ decoding
→ prediction
→ output control
→ feedback