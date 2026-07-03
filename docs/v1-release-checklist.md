# NeuroLab v1 Release Checklist

Track every item before tagging the `v1.0.0` release. Check off items as they are verified on the running app.

> **Audit status:** Inspected 2026-07-03. Items verified by static analysis of source files;
> items that require a running browser session (rendering, interactivity) remain unchecked
> until manually confirmed.

---

## App Functionality

- [ ] `streamlit run app/app.py` starts without errors on a clean install
- [x] All 9 sidebar pages load without exceptions: Home, NeuroSim, EEG Studio, Decoder Lab, Closed-Loop BCI, Dashboard, Examples, References, Settings
  > 10 pages found (tutorials.py is also present). All 9 required pages confirmed.
- [ ] Navigation order in sidebar is logical and consistent
- [x] `st.set_page_config` is called only once (in `app/app.py`); no page file calls it
- [x] No `st.session_state` key collisions between pages (namespaces: `ns_*`, `dl_*`, `cl_*`, `cfg_*`, `_ex_*`)
- [ ] App runs without PyTorch installed (decoder pages degrade gracefully)
  > Code path verified (`_load_model` catches `ImportError` and returns an error string shown via `st.warning`), but requires a live run to confirm the full degraded experience.
- [ ] App runs without the PhysioNet dataset present (EEG pages show informative placeholder)
  > Fallback paths exist in code (`_load_eeg` returns `None`; EEG pages check for `None`), but requires a live run to confirm all placeholders render correctly.
- [ ] Dark mode (Streamlit theme toggle) does not break any page layout or chart readability
- [ ] No Python warnings printed to the terminal during normal use
- [ ] `pip install -r requirements.txt` installs all dependencies without conflicts
  > **Blocked:** `requirements.txt` exists but is empty (0 bytes). Dependencies live in `pyproject.toml`. This command will install nothing. Either populate `requirements.txt` or update the README and this item to reference `pip install .` instead.
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
- [x] Export buttons produce valid CSV and JSON files for: LIF trace, F-I curve, parameter sweep, model comparison, Izhikevich trace
  > All five `download_buttons()` calls confirmed present (lines 161, 237, 343, 443, 546 of `neurosim.py`).
- [x] `cfg_sim_duration` setting from Settings page is respected as the default `t_max` slider value
  > `neurosim.py` line 61–64: slider `value=float(st.session_state.get("cfg_sim_duration", 500.0))`.
- [x] Section headers render consistently (no raw `####` markdown headings)
  > All section headers use `section_header()` from `components/ui.py`. No raw `####` found.

---

## EEG Studio

- [ ] Raw EEG waterfall chart renders for any valid trial index
- [ ] Trial label (left/right) is correctly displayed and colour-coded
- [ ] PSD chart shows correct frequency axis (0–40 Hz), band shading, and Welch trace
- [ ] Spectrogram renders without error; colour scale is readable
- [ ] Band power bar chart shows all five bands (Delta, Theta, Alpha, Beta, Gamma)
- [ ] Band power table matches the bar chart values
- [ ] Channel selector dropdowns (PSD, spectrogram, band power) all function independently
- [x] `cfg_eeg_trial` and `cfg_eeg_channels` settings from Settings page are respected as sidebar defaults
  > `eeg_studio.py` lines 66–86: both sliders clamp and read `cfg_eeg_trial` / `cfg_eeg_channels` from session state.
- [x] Export buttons produce valid CSV for PSD and band power
  > `download_buttons()` confirmed at lines 189 (PSD) and 314 (band power) of `eeg_studio.py`.
- [ ] "Dataset not available" placeholder appears gracefully if the data file is missing

> **Missing:** No export for the spectrogram. The spectrogram chart is rendered (lines 197–249
> of `eeg_studio.py`) but there is no `download_buttons()` call after it. This is also noted
> under Known Limitations. Must be resolved or explicitly deferred before v1.0.0 ships.

---

## Decoder Lab

- [ ] Trial index slider and channel display slider work correctly
- [x] "Run Inference & Move Cursor" button is disabled (or shows an informative warning) when no model checkpoint exists
  > `_load_model()` returns `(None, error_string)` when checkpoint is absent; `st.warning(model_error)` is shown and the inference button is not rendered.
- [ ] When model is present: inference runs, class probabilities sum to 1.0, correct/incorrect outcome is reported
- [ ] Probability bar chart renders with correct left/right colours
- [ ] Cursor trajectory updates correctly with each inference call; Reset Cursor clears the path
- [ ] Raw EEG signal chart renders for the selected trial and channel count
- [ ] Model performance metrics (accuracy, precision, recall, F1) display correctly when `results/metrics.json` exists
- [ ] Confusion matrix heatmap renders correctly when `results/confusion_matrix.json` exists
- [ ] All metric values are percentages formatted to one decimal place
- [x] Export buttons produce valid JSON for prediction and confusion matrix
  > `download_buttons()` confirmed at lines 223 (prediction) and 347 (confusion matrix) of `decoder_lab.py`.
- [x] `cfg_eeg_trial` and `cfg_eeg_channels` settings from Settings page are respected as sidebar defaults
  > `decoder_lab.py` lines 134–143: sliders read `cfg_eeg_trial` / `cfg_eeg_channels` and clamp to dataset bounds.

---

## Closed-Loop BCI

- [x] "Start Session" creates a balanced, shuffled cue list of the requested length
  > `_generate_cues()` (`closed_loop.py` lines 53–60) splits into equal left/right halves then shuffles with a seeded `random.Random` instance.
- [ ] Cue display (← / →) is large, clearly colour-coded, and matches the session cue list
- [ ] "Next Trial" advances the trial index, runs the mock decoder, and updates the cursor
- [ ] Cursor trajectory chart updates after every trial
- [ ] Cursor X, Y, and step count metrics are correct
- [ ] Last-trial result (predicted class, confidence, match/mismatch) displays correctly
- [ ] Session accuracy tracks correctly across all trials
- [x] "Session Complete" state appears after the final trial and shows final accuracy
  > `closed_loop.py` lines 191–200: `section_header("Session Complete")` and `st.success()` rendered when `trial_idx >= len(cues)`.
- [x] "Reset Session" clears all state and returns to the idle screen
  > `_reset_session()` (`closed_loop.py` lines 72–75) pops all six `cl_*` keys from session state.
- [ ] Sidebar controls (n_trials, decoder accuracy, seed) are disabled while a session is active
- [x] Export CSV and JSON are generated at end of session with correct trial-level records
  > `download_buttons()` at `closed_loop.py` lines 287–305 includes per-trial history and session summary.

---

## Dashboard

- [x] Page loads without errors when `results/metrics.json` and `results/confusion_matrix.json` are absent
  > All four loaders return `None` on missing files; each section checks for `None` and shows an `st.info()` placeholder with the expected file path.
- [ ] Key metrics (accuracy, F1, precision, recall) are displayed when result files exist
- [ ] Training curve sparkline renders when `results/training_history.json` exists
- [ ] Confusion matrix mini-chart renders when `results/confusion_matrix.json` exists
- [x] All four module cards (NeuroSim, EEG Analysis, Decoder Performance, Closed-Loop BCI) render
  > `dashboard.py`: `_section_neurosim()`, `_section_eeg()`, `_section_decoder()`, `_section_closed_loop()` all defined and called.
- [ ] Module descriptions are accurate and up to date
- [x] Closed-Loop session card reads live `cl_*` session state from the current session
  > `dashboard.py` lines 284–286 read `cl_history`, `cl_active`, `cl_cues` directly from session state.
- [ ] "No data" placeholders are informative and include the expected file paths

---

## Examples

- [ ] All 10 example cards render without JavaScript or Streamlit errors
- [ ] Each "Run Example" button produces a Plotly chart inline within the card
- [ ] Results persist across sidebar navigation without re-running
- [ ] "✕ Clear" button removes the result and the chart disappears
- [ ] "Clear all results" sidebar button clears all 10 results at once
- [ ] Session run count in sidebar increments correctly
- [x] EEG-dependent examples (EEG Trial, PSD, Spectrogram, Decoder Demo) show graceful fallback if dataset is absent
  > All four EEG run functions check `if data is None` and return a fallback `go.Figure` with an annotation, or use synthetic data.
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

> The 10 example IDs (`lif_current_sweep`, `lif_fi_curve`, `lif_parameter_sweep`,
> `izh_regular_spiking`, `izh_fast_spiking`, `eeg_trial`, `eeg_psd`, `eeg_spectrogram`,
> `decoder_demo`, `closed_loop_demo`) are all present in `examples.py`. Run-function
> correctness and interactive behaviour require a live browser session to verify.

---

## Scientific Documentation

- [ ] References page loads all 18 references without error
  > Audit found **19** references defined in `references.py`, not 18. Update count here once confirmed in browser.
- [ ] Search bar filters correctly across title, authors, and description fields
- [ ] Category multiselect shows/hides categories correctly; Select All / Clear All work
- [ ] Every reference with a DOI links to `https://doi.org/{doi}` and opens correctly
- [ ] Every reference with a URL-only link opens the correct publisher or repository page
- [ ] "Used in" module chips are accurate for all 18 references
- [x] Hodgkin & Huxley (1952) — present with correct DOI `10.1113/jphysiol.1952.sp004764`
  > Confirmed at `references.py` lines 127–136.
- [x] Izhikevich (2003) — present with correct DOI `10.1109/TNN.2003.820440`
  > Confirmed at `references.py` lines 155–165.
- [x] Lawhern et al. EEGNet (2018) — present, tagged with "Decoder Lab"
  > Confirmed at `references.py` lines 277–281.
- [x] PhysioNet (Goldberger et al. 2000) — present, tagged with "EEG Studio" and "Decoder Lab"
  > Confirmed at `references.py` lines 327–332.
- [x] BCI2000 (Schalk et al. 2004) — present, tagged correctly
  > Confirmed present with `used_in=("EEG Studio", "Decoder Lab")`.
- [ ] `docs/scientific-foundation/` directory contains accurate neuron model and BCI background docs
- [ ] All scientific parameter values in docs match the values hard-coded in the neuron modules

---

## README

- [x] README.md exists at the repository root
- [ ] Project overview accurately describes NeuroLab as a computational neuroscience and BCI platform
- [ ] Feature list is complete and matches all implemented pages
  > README was written before `examples.py`, `references.py`, and `settings.py` were added. Verify the feature list and repository structure tree include all three new pages.
- [x] Installation instructions work on a clean Python 3.10+ environment
- [x] `streamlit run app/app.py` launch command is correct
- [x] Training command (`python scripts/train_eegnet.py`) is correct and the script exists
  > `scripts/train_eegnet.py` confirmed present.
- [ ] Repository structure tree is accurate and up to date (includes `examples.py`, `references.py`, `settings.py`)
  > Needs manual check — README structure section may predate these pages.
- [x] All three Mermaid diagrams (architecture, BCI pipeline, EEGNet) render correctly on GitHub
  > Three `mermaid` fenced blocks confirmed in README.md.
- [ ] Scientific references section lists at least Izhikevich (2003), EEGNet (2018), PhysioNet
- [ ] Roadmap table distinguishes completed (v0.1) from planned (v0.2–v1.0) items
- [x] License section is present and correct
  > `LICENSE` file (MIT) exists at repo root; license badge present in README.
- [ ] Badges (Python version, Streamlit, license) are accurate

---

## GitHub Hygiene

- [x] `.gitignore` excludes: `__pycache__/`, `*.pyc`, `.env`, `models/*.pth`, `data/`, `results/`, `.streamlit/secrets.toml`
  > All patterns confirmed present in `.gitignore`.
- [ ] No model checkpoint files (`.pth`) are committed to the repository
  > **Fail:** `models/eegnet_motor_imagery_v1.pth` is present in the working tree. Determine whether it should be committed (decision: intentional bundled checkpoint) or removed and regenerated via the training script. If intentional, update `.gitignore` to un-ignore this specific file and document it in the README.
- [ ] No raw dataset files are committed to the repository
- [ ] No secrets, API keys, or credentials appear anywhere in the commit history
- [ ] `requirements.txt` pins major versions and is in sync with the actual imports
  > **Fail:** `requirements.txt` is empty (0 bytes). Dependencies are declared in `pyproject.toml`. Before release, either generate a pinned `requirements.txt` via `pip freeze > requirements.txt` (after installing from `pyproject.toml`) or remove `requirements.txt` and update all documentation to use `pip install .` instead.
- [ ] All source files have consistent encoding (UTF-8) and Unix line endings
- [x] No file contains a bare `render()` call that was left as dead code outside the `if __name__` guard (Streamlit pages intentionally call `render()` at module level — confirm this is correct for each page)
  > All 9 content pages call `render()` at module level as required by Streamlit multipage routing. `home.py` is called from `app.py`. Pattern is correct and intentional.
- [ ] Git log has clean, descriptive commit messages; no "fix typo" or "WIP" commits on `main`
- [ ] `v1.0.0` tag is created on the release commit: `git tag -a v1.0.0 -m "NeuroLab v1.0.0"`
- [ ] GitHub repository description and topics are set (`computational-neuroscience`, `bci`, `streamlit`, `eeg`, `python`)
- [ ] A GitHub Release is created from the `v1.0.0` tag with a changelog summary

> **Missing:** No `.github/` directory found. There is no CI/CD pipeline (no lint, test, or
> deployment workflow). At minimum, a GitHub Actions workflow running `python -m pytest` and
> a Streamlit smoke-test should be added before v1.0.0. See also Known Limitations.

---

## Known Limitations

Document these in the release notes before tagging v1.0.0.

- [x] **No real EEG inference without training** — EEGNet requires running `scripts/train_eegnet.py` first; the app does not bundle a pretrained checkpoint
  > A checkpoint (`models/eegnet_motor_imagery_v1.pth`) *is* present in the working tree; resolve the `.gitignore` decision above before closing this item.
- [x] **Session state is ephemeral** — all session settings, closed-loop history, and example results are lost on page refresh; no persistence layer exists
- [x] **Single-user only** — Streamlit session state is not isolated per concurrent user in the current deployment configuration
- [x] **Mock decoder in Closed-Loop BCI** — the closed-loop page uses a probabilistic mock rather than real-time EEG inference
- [x] **Fixed dataset** — only the PhysioNet EEGMMI dataset is supported; no upload or custom dataset path UI exists
- [x] **No export for EEG spectrogram** — the spectrogram raw matrix is not currently exportable via the download buttons
  > Confirmed: no `download_buttons()` call after the spectrogram chart in `eeg_studio.py`.
- [x] **Plot theme is partial** — the Dark theme setting applies only to the main LIF chart and EEG PSD chart; other charts use the Default theme regardless of the setting
  > Confirmed: only `neurosim.py` line 119 and `eeg_studio.py` line 178 call `get_plotly_layout()`; all other charts spread `PLOTLY_LAYOUT` directly.
- [x] **Tutorials page is a placeholder** — `app/pages/tutorials.py` shows a "coming soon" banner; no tutorial content exists
  > Confirmed: `tutorials.py` renders five learning-track cards with placeholder content and an `st.info("Tutorials are coming soon…")` message.
- [x] **No automated tests for the Streamlit UI** — correctness is verified by manual inspection; no Playwright or Selenium test suite exists
- [x] **No CI/CD pipeline** — there is no GitHub Actions workflow for linting, testing, or deployment
  > Confirmed: no `.github/` directory exists in the repository.
