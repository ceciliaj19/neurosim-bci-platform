# NeuroLab v1 Release Checklist

Track every item before tagging the `v1.0.0` release. Check off items as they are verified on the running app.

---

## App Functionality

- [ ] `streamlit run app/app.py` starts without errors on a clean install
- [ ] All 9 sidebar pages load without exceptions: Home, NeuroSim, EEG Studio, Decoder Lab, Closed-Loop BCI, Dashboard, Examples, References, Settings
- [ ] Navigation order in sidebar is logical and consistent
- [ ] `st.set_page_config` is called only once (in `app/app.py`); no page file calls it
- [ ] No `st.session_state` key collisions between pages (namespaces: `ns_*`, `dl_*`, `cl_*`, `cfg_*`, `_ex_*`)
- [ ] App runs without PyTorch installed (decoder pages degrade gracefully)
- [ ] App runs without the PhysioNet dataset present (EEG pages show informative placeholder)
- [ ] Dark mode (Streamlit theme toggle) does not break any page layout or chart readability
- [ ] No Python warnings printed to the terminal during normal use
- [ ] `pip install -r requirements.txt` installs all dependencies without conflicts
- [ ] Python 3.10+ compatibility confirmed (uses `match`, `str | None` union syntax)

---

## NeuroSim

- [ ] LIF neuron simulator renders membrane potential trace for all slider positions
- [ ] Spike markers appear on the trace at correct times
- [ ] Threshold line is visible and matches the `v_threshold` slider value
- [ ] F-I curve computes and plots without error; curve is monotonically non-decreasing above rheobase
- [ ] Parameter sweep runs for all three sweepable parameters (τm, resistance, v_threshold)
- [ ] Model comparison renders both LIF and Izhikevich (RS) traces on a shared time axis
- [ ] All six Izhikevich presets load and simulate without error
- [ ] Izhikevich trace shows qualitatively correct firing pattern for each preset (RS, FS, IB, CH, LTS, RZ)
- [ ] Export buttons produce valid CSV and JSON files for: LIF trace, F-I curve, parameter sweep, model comparison, Izhikevich trace
- [ ] `cfg_sim_duration` setting from Settings page is respected as the default `t_max` slider value
- [ ] Section headers render consistently (no raw `####` markdown headings)

---

## EEG Studio

- [ ] Raw EEG waterfall chart renders for any valid trial index
- [ ] Trial label (left/right) is correctly displayed and colour-coded
- [ ] PSD chart shows correct frequency axis (0–40 Hz), band shading, and Welch trace
- [ ] Spectrogram renders without error; colour scale is readable
- [ ] Band power bar chart shows all five bands (Delta, Theta, Alpha, Beta, Gamma)
- [ ] Band power table matches the bar chart values
- [ ] Channel selector dropdowns (PSD, spectrogram, band power) all function independently
- [ ] `cfg_eeg_trial` and `cfg_eeg_channels` settings from Settings page are respected as sidebar defaults
- [ ] Export buttons produce valid CSV for PSD and band power
- [ ] "Dataset not available" placeholder appears gracefully if the data file is missing

---

## Decoder Lab

- [ ] Trial index slider and channel display slider work correctly
- [ ] "Run Inference & Move Cursor" button is disabled (or shows an informative warning) when no model checkpoint exists
- [ ] When model is present: inference runs, class probabilities sum to 1.0, correct/incorrect outcome is reported
- [ ] Probability bar chart renders with correct left/right colours
- [ ] Cursor trajectory updates correctly with each inference call; Reset Cursor clears the path
- [ ] Raw EEG signal chart renders for the selected trial and channel count
- [ ] Model performance metrics (accuracy, precision, recall, F1) display correctly when `results/metrics.json` exists
- [ ] Confusion matrix heatmap renders correctly when `results/confusion_matrix.json` exists
- [ ] All metric values are percentages formatted to one decimal place
- [ ] Export buttons produce valid JSON for prediction and confusion matrix
- [ ] `cfg_eeg_trial` and `cfg_eeg_channels` settings from Settings page are respected as sidebar defaults

---

## Closed-Loop BCI

- [ ] "Start Session" creates a balanced, shuffled cue list of the requested length
- [ ] Cue display (← / →) is large, clearly colour-coded, and matches the session cue list
- [ ] "Next Trial" advances the trial index, runs the mock decoder, and updates the cursor
- [ ] Cursor trajectory chart updates after every trial
- [ ] Cursor X, Y, and step count metrics are correct
- [ ] Last-trial result (predicted class, confidence, match/mismatch) displays correctly
- [ ] Session accuracy tracks correctly across all trials
- [ ] "Session Complete" state appears after the final trial and shows final accuracy
- [ ] "Reset Session" clears all state and returns to the idle screen
- [ ] Sidebar controls (n_trials, decoder accuracy, seed) are disabled while a session is active
- [ ] Export CSV and JSON are generated at end of session with correct trial-level records

---

## Dashboard

- [ ] Page loads without errors when `results/metrics.json` and `results/confusion_matrix.json` are absent
- [ ] Key metrics (accuracy, F1, precision, recall) are displayed when result files exist
- [ ] Training curve sparkline renders when `results/training_history.json` exists
- [ ] Confusion matrix mini-chart renders when `results/confusion_matrix.json` exists
- [ ] All four module cards (NeuroSim, EEG Analysis, Decoder Performance, Closed-Loop BCI) render
- [ ] Module descriptions are accurate and up to date
- [ ] Closed-Loop session card reads live `cl_*` session state from the current session
- [ ] "No data" placeholders are informative and include the expected file paths

---

## Examples

- [ ] All 10 example cards render without JavaScript or Streamlit errors
- [ ] Each "Run Example" button produces a Plotly chart inline within the card
- [ ] Results persist across sidebar navigation without re-running
- [ ] "✕ Clear" button removes the result and the chart disappears
- [ ] "Clear all results" sidebar button clears all 10 results at once
- [ ] Session run count in sidebar increments correctly
- [ ] EEG-dependent examples (EEG Trial, PSD, Spectrogram, Decoder Demo) show graceful fallback if dataset is absent
- [ ] LIF Current Sweep — trace and spike markers are correct
- [ ] LIF F-I Curve — curve rises from zero above rheobase
- [ ] LIF Parameter Sweep — firing rate decreases as τm increases (at fixed I=2.0)
- [ ] Izhikevich Regular Spiking — shows spike-rate adaptation
- [ ] Izhikevich Fast Spiking — shows sustained non-adapting high-frequency firing
- [ ] EEG Trial Exploration — 8-channel waterfall loads and is readable
- [ ] EEG Power Spectrum — alpha and beta peaks are visible; band annotations are correct
- [ ] EEG Spectrogram — time-frequency heatmap renders without blank panels
- [ ] Decoder Prediction Demo — probability bar sums to 100 %; outcome label is correct
- [ ] Closed-Loop Cursor Demo — trajectory plot shows 12 steps from origin

---

## Scientific Documentation

- [ ] References page loads all 18 references without error
- [ ] Search bar filters correctly across title, authors, and description fields
- [ ] Category multiselect shows/hides categories correctly; Select All / Clear All work
- [ ] Every reference with a DOI links to `https://doi.org/{doi}` and opens correctly
- [ ] Every reference with a URL-only link opens the correct publisher or repository page
- [ ] "Used in" module chips are accurate for all 18 references
- [ ] Hodgkin & Huxley (1952) — present with correct DOI `10.1113/jphysiol.1952.sp004764`
- [ ] Izhikevich (2003) — present with correct DOI `10.1109/TNN.2003.820440`
- [ ] Lawhern et al. EEGNet (2018) — present, tagged with "Decoder Lab"
- [ ] PhysioNet (Goldberger et al. 2000) — present, tagged with "EEG Studio" and "Decoder Lab"
- [ ] BCI2000 (Schalk et al. 2004) — present, tagged correctly
- [ ] `docs/scientific-foundation/` directory contains accurate neuron model and BCI background docs
- [ ] All scientific parameter values in docs match the values hard-coded in the neuron modules

---

## README

- [ ] README.md exists at the repository root
- [ ] Project overview accurately describes NeuroLab as a computational neuroscience and BCI platform
- [ ] Feature list is complete and matches all implemented pages
- [ ] Installation instructions work on a clean Python 3.10+ environment
- [ ] `streamlit run app/app.py` launch command is correct
- [ ] Training command (`python scripts/train_eegnet.py`) is correct and the script exists
- [ ] Repository structure tree is accurate and up to date (includes `examples.py`, `references.py`, `settings.py`)
- [ ] All three Mermaid diagrams (architecture, BCI pipeline, EEGNet) render correctly on GitHub
- [ ] Scientific references section lists at least Izhikevich (2003), EEGNet (2018), PhysioNet
- [ ] Roadmap table distinguishes completed (v0.1) from planned (v0.2–v1.0) items
- [ ] License section is present and correct
- [ ] Badges (Python version, Streamlit, license) are accurate

---

## GitHub Hygiene

- [ ] `.gitignore` excludes: `__pycache__/`, `*.pyc`, `.env`, `models/*.pth`, `data/`, `results/`, `.streamlit/secrets.toml`
- [ ] No model checkpoint files (`.pth`) are committed to the repository
- [ ] No raw dataset files are committed to the repository
- [ ] No secrets, API keys, or credentials appear anywhere in the commit history
- [ ] `requirements.txt` pins major versions and is in sync with the actual imports
- [ ] All source files have consistent encoding (UTF-8) and Unix line endings
- [ ] No file contains a bare `render()` call that was left as dead code outside the `if __name__` guard (Streamlit pages intentionally call `render()` at module level — confirm this is correct for each page)
- [ ] Git log has clean, descriptive commit messages; no "fix typo" or "WIP" commits on `main`
- [ ] `v1.0.0` tag is created on the release commit: `git tag -a v1.0.0 -m "NeuroLab v1.0.0"`
- [ ] GitHub repository description and topics are set (`computational-neuroscience`, `bci`, `streamlit`, `eeg`, `python`)
- [ ] A GitHub Release is created from the `v1.0.0` tag with a changelog summary

---

## Known Limitations

Document these in the release notes before tagging v1.0.0.

- [ ] **No real EEG inference without training** — EEGNet requires running `scripts/train_eegnet.py` first; the app does not bundle a pretrained checkpoint
- [ ] **Session state is ephemeral** — all session settings, closed-loop history, and example results are lost on page refresh; no persistence layer exists
- [ ] **Single-user only** — Streamlit session state is not isolated per concurrent user in the current deployment configuration
- [ ] **Mock decoder in Closed-Loop BCI** — the closed-loop page uses a probabilistic mock rather than real-time EEG inference
- [ ] **Fixed dataset** — only the PhysioNet EEGMMI dataset is supported; no upload or custom dataset path UI exists
- [ ] **No export for EEG spectrogram** — the spectrogram raw matrix is not currently exportable via the download buttons
- [ ] **Plot theme is partial** — the Dark theme setting applies only to the main LIF chart and EEG PSD chart; other charts use the Default theme regardless of the setting
- [ ] **Tutorials page is a placeholder** — `app/pages/tutorials.py` shows a "coming soon" banner; no tutorial content exists
- [ ] **No automated tests for the Streamlit UI** — correctness is verified by manual inspection; no Playwright or Selenium test suite exists
- [ ] **No CI/CD pipeline** — there is no GitHub Actions workflow for linting, testing, or deployment
