# Breast cell classification — educational ML demo

A deployment-ready Streamlit application built from the original Gaussian Naive Bayes project. Visitors can try two public dataset examples or enter all 30 numerical measurements, edit values, and run classification. **This is not an image detector.** It does not accept photographs, scans, or patient documents.

> **Educational demo only — not for medical diagnosis or treatment.** Predictions can be wrong. Do not use this app for healthcare decisions, screening, or individual risk estimation. A benign model prediction does not establish that someone is healthy.

## Status

Prepared and tested locally on **4 November 2026**, using **Python 3.12.16**. Source code is public at **[palaksharma3/breast-cell-classification-demo](https://github.com/palaksharma3/breast-cell-classification-demo)** on branch **`main`**. The app is **deployment-ready, not publicly deployed**: there is no public Streamlit app URL yet. Nothing was posted to LinkedIn or a website. The local URL is not accessible to other people over the internet.

## Run the prepared app on this Windows computer

Open PowerShell and run:

```powershell
Set-Location 'C:\Users\Palak Sharma\Desktop\AIML\Cancer cell detection'
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py --server.address 127.0.0.1
```

Open <http://localhost:8501>. If the preview is already running, just open the URL; do not start a second copy. Stop a server you started in a terminal with **Ctrl+C**. No activation script, API key, account, model download, or secret is required for local use.

The prepared `.venv` uses an isolated Python 3.12 interpreter installed with `uv`, rather than the computer's default Python 3.15. To reinstall its packages if needed:

```powershell
python -m uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

## Fresh Windows setup (another computer)

1. Install **Python 3.12** from <https://www.python.org/downloads/> with the Python launcher enabled.
2. Download or clone this project's source into a folder, then open PowerShell in that folder.
3. Run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py --server.address 127.0.0.1
```

Use Python 3.12, not an arbitrary newest release. Streamlit Community Cloud uses Linux, so no Windows executable or virtual environment is uploaded.

## Try it

1. On **Try the model**, leave **Public example A** selected.
2. Expand the measurement sections if desired, then select **Run classification**.
3. Try **Public example B**, or change a value and run again.
4. For your own non-sensitive educational data, select **Enter my own measurements**. All 30 fields start blank and are required.
5. Open **Model & evidence** to inspect actual evaluation metrics, confusion counts, class mapping, and limitations.

The form groups features into mean, standard error, and worst measurements. “Worst” means the mean of the three largest values. The numbers must use the feature definitions and units in the source dataset; these are not everyday health measurements. The app does not extract features from images. It has no CSV upload or batch prediction interface.

Each input is matched by its exact dataset feature name, then reordered to `load_breast_cancer().feature_names`. Blank, non-numeric, non-finite, negative/out-of-range values and missing/extra features are rejected by the reusable model layer. Zero is accepted only where observed in the public dataset. The interface deliberately accepts only each feature's observed dataset minimum-to-maximum range: these are **demo bounds, not clinical reference limits**. Plausible inputs outside those bounds are intentionally unsupported. Bounds come from all public rows solely for interface validation, not fitted preprocessing. There is no silent imputation or invented custom-case default.

## Genuine model evaluation

- Dataset: Wisconsin Diagnostic Breast Cancer, bundled with scikit-learn.
- 569 rows, 30 numerical features; 212 malignant and 357 benign.
- Mapping: **0 = malignant; 1 = benign**.
- Algorithm: `GaussianNB()` with default parameters.
- Split: `train_test_split(data.data, data.target, test_size=0.33, random_state=42)`.
- **Original unstratified split retained**: 381 training / 188 test rows. No scaling, tuning, or stratification was added.
- Training is deterministic and cached at app startup. No unsafe serialized model file or external training download is used.

| Held-out metric | Result |
|---|---:|
| Accuracy | 94.15% (177 / 188) |
| Malignant precision | 92.42% (61 / 66) |
| Malignant recall | 91.04% (61 / 67) |

| Actual / predicted | Malignant (0) | Benign (1) |
|---|---:|---:|
| Malignant (0) | 61 | 6 |
| Benign (1) | 5 | 116 |

There are **6 missed malignant labels** and **5 benign labels classified as malignant** in this split. These results come from actual held-out predictions, not the unrelated five-label accuracy calculation in `main.py`. The original file is preserved as historical project code; the new application does not import it or display that misleading calculation.

### Limitations and provenance

A small historical dataset and one split do not establish clinical reliability, generalization to other populations or measurement instruments, or fairness. GaussianNB assumes conditional independence despite correlated features. No external, prospective, clinical, or fairness validation has been performed. Public example rows 0 (recorded malignant) and 19 (recorded benign) belong to the same dataset used for training/testing and are **not independent validation cases**. Model class labels are not calibrated cancer-risk estimates.

Data origin: features extracted from digitized fine needle aspirate images of breast masses. Source: Wolberg, Mangasarian, Street & Street (1993), *Breast Cancer Wisconsin (Diagnostic)*, UCI Machine Learning Repository, DOI [10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B), dataset license **CC BY 4.0**. Retain attribution when sharing.

- [scikit-learn dataset documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)
- [UCI dataset, original feature definitions, and attribution](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)

## Make it public today: GitHub → Streamlit Community Cloud

Streamlit Community Cloud is a practical route for this small Streamlit demo, subject to its current free-tier limits and account terms. The approved public GitHub repository is already prepared. These remaining steps require **you** to approve the hosting connection and deploy the app; GitHub publication alone does not host a running application.

### 1. Use the prepared public repository

1. Open **[palaksharma3/breast-cell-classification-demo](https://github.com/palaksharma3/breast-cell-classification-demo)**. The application files are on branch **`main`**; no additional repository creation or upload is needed.
2. The publication includes **only** these application files, preserving the folder structure:

```text
streamlit_app.py
model.py
requirements.txt
README.md
.gitignore
.streamlit/config.toml
tests/test_app.py
tests/test_model.py
```

The above paths are GitHub/Linux paths; locally, `.streamlit\config.toml` and `tests\test_app.py` are Windows paths. In GitHub's web upload screen, dot-folders can be easy to miss: verify `.streamlit/config.toml` exists afterward. The app will still start without the theme file, but include it for the intended appearance and settings.

Do **not** upload `.venv`, `__pycache__`, `.env` files, secrets, logs, or identifiable patient records. `.gitignore` protects normal Git workflows, but is not a substitute for checking browser uploads. The scratch `tempCodeRunnerFile.py` remains untouched locally and is ignored. The original `main.py` may be kept privately as historical work; the tested deployment file list intentionally excludes it so reviewers use the correct app and metrics.

### 2. Deploy on Streamlit Community Cloud

1. Open <https://share.streamlit.io> and sign in. Connect the intended GitHub account yourself; review any requested permissions.
2. Select **Create app** (or the equivalent new-app button), then deploy from GitHub.
3. Select **`palaksharma3/breast-cell-classification-demo`**.
4. Set **Branch** to **`main`**.
5. Set **Main file path / entrypoint** to **`streamlit_app.py`**, not `main.py`.
6. Under **Advanced settings**, choose **Python 3.12**. No secrets are needed.
7. Select **Deploy**. The host installs the root `requirements.txt` and trains the small baseline at startup.
8. After it starts, go to **Settings → Sharing** and ensure **“This app is public and searchable”** is selected if you want everyone to access it. Review this visibility decision before saving.
9. Copy the **actual** `https://...streamlit.app` URL shown by the host. Do not use localhost, a GitHub code URL, or an invented app URL.

Official guide: <https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy>.

### 3. Verify before sharing

- Open the actual app URL in an **incognito/private window while signed out**. It must load without requesting a login.
- Run both examples and verify that a result appears without an exception.
- Try a blank custom case and confirm a useful error appears.
- Check the evidence tab and educational notice on desktop and phone.
- If it is private, revisit Sharing. If it will not build, inspect deployment logs, verify the entrypoint and Python 3.12, and check that all required files are on the selected branch.
- Free hosting may sleep when idle and cold-start on the next visit. This is not a guaranteed-uptime medical service. Once verification succeeds, then describe it as **live**.

### 4. Add it to your portfolio

- **CV:** use the truthful bullet below now. Add the [public repository link](https://github.com/palaksharma3/breast-cell-classification-demo) now; add a public demo URL only after deployment and the incognito check pass.
- **LinkedIn:** add a project or Featured link to the actual live URL and, optionally, the repository. No post has been submitted automatically.
- **Website:** add a “View demo” link to the real URL and a “Source code” link to the real repository. A normal link works without an iframe or changes to your website's infrastructure. Your website has not been changed.

## Ready-to-use CV / portfolio wording

**Project title:** Breast Cell Classification — Gaussian Naive Bayes & Streamlit

**CV bullet (accurate before public deployment):**

> Built and tested a deployment-ready Streamlit ML demo with 30 validated numerical inputs and public-dataset examples; preserved a reproducible Gaussian Naive Bayes baseline achieving 94.15% accuracy on a 188-sample held-out split, with transparent confusion-matrix reporting and educational-use safeguards.

**LinkedIn / website summary (before deployment):**

> I built an interactive educational breast-cell classification demo using Python, scikit-learn, and Streamlit. Users can explore public Wisconsin dataset examples or edit 30 numerical features. The app includes input validation, reproducible held-out evaluation, and clear model limitations. It is prepared for Streamlit Community Cloud deployment. This is a portfolio ML demonstration, not a medical diagnostic tool or an image classifier.

Source code: <https://github.com/palaksharma3/breast-cell-classification-demo>.

After public **app** deployment and verification, replace “prepared for Streamlit Community Cloud deployment” with “available online” and append the real demo URL. Do not call it clinically validated, claim diagnostic accuracy, or imply it processes photographs.

## Privacy and configuration

Submitted values are processed server-side in memory. Application code does not intentionally write or log user measurements; no database, analytics integration, or external prediction service is configured. The local configuration opts out of Streamlit usage-stat collection (`browser.gatherUsageStats = false`), but Streamlit Community Cloud overrides this setting to `true`; it is not a cloud telemetry opt-out guarantee ([official status documentation](https://docs.streamlit.io/deploy/streamlit-community-cloud/status)). The hosting platform may retain access/operational logs; **do not enter identifiable patient data or sensitive measurements**. Do not promise client-only processing or zero host logging. Keep CORS and XSRF protections enabled. Local preview binds to loopback; let Community Cloud manage its own network binding.

## Tests and verification record

```powershell
Set-Location 'C:\Users\Palak Sharma\Desktop\AIML\Cancer cell detection'
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

**Result on 4 November 2026: 16 tests passed (`Ran 16 tests ... OK`).** Tests use standard-library unittest and Streamlit's bundled AppTest; no separate testing package is required. Coverage includes exact feature ordering even with reversed dictionaries; deterministic training; actual held-out evaluation; class mapping; missing/extra, blank, non-finite, non-numeric, and out-of-range values; all 569 public rows; initial app rendering; both examples; editing; complete custom input; invalid custom input; and clearing results when changing sources. AppTest may print a benign `missing ScriptRunContext` warning when run outside the server.

Static UI detector: no findings. Real server verification returned **HTTP 200** for both `http://127.0.0.1:8501/` and `http://127.0.0.1:8501/_stcore/health` (body: `ok`). All 40 installed packages passed `uv pip check`. The local preview was also opened in the browser canvas. The README documents stable commands rather than embedding a transient process ID.

## Files and original preservation

- `streamlit_app.py`: native Streamlit interface, examples, editable form, evidence, disclosures.
- `model.py`: reusable deterministic training, validation, ordered inference, metrics.
- `requirements.txt`: exact direct dependency pins; transitive dependencies resolved by the installer.
- `.streamlit\config.toml`: accessible light theme, telemetry preference, security settings.
- `tests\test_model.py`, `tests\test_app.py`: model and UI tests.
- `.gitignore`: excludes local environments, secrets, caches, scratch file, and logs.
- `README.md`: setup, deployment, verification, limitations, and portfolio wording.
- `.venv`: local-only prepared Python 3.12 environment, never publish it.
- `main.py`, `tempCodeRunnerFile.py`: originals preserved, not part of the deployed app.

Original `main.py` SHA-256 before changes:

```text
2FECC242F84E72F1C5CC4A3780D3973AF6BB5A429F82F1DB04A9418622F749FE
```

Check preservation at any time:

```powershell
Get-FileHash .\main.py -Algorithm SHA256
```

The hash was compared again after implementation. Git was initialized for the approved source publication with remote `origin` at <https://github.com/palaksharma3/breast-cell-classification-demo.git> and branch `main`. Only the eight allowlisted application, test, configuration, and documentation files are tracked. Both original scripts and `.venv` remain local and ignored. No hosting account permissions or cloud deployment were performed.
