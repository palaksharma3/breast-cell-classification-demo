"""Native Streamlit workbench: examples first, editable features, honest evidence.

Operate surface: a restrained light theme for recruiters exploring on a laptop.
Native controls retain keyboard access and responsive layout; no decorative CSS.
The primary task is to inspect an example, edit measurements, and run the model.
"""

import pandas as pd
import streamlit as st

from model import (
    CLASS_NAMES, DATA, EXAMPLES, FEATURE_NAMES, LOWER_BOUNDS, UPPER_BOUNDS,
    example_values, predict_measurements, train_model,
)

st.set_page_config(page_title="Breast Cell Classification | ML Demo", page_icon="🔬", layout="centered")
st.title("Breast cell classification")
st.write("Explore a Gaussian Naive Bayes model using 30 numerical measurements of cell nuclei.")
st.warning("Educational demo — not for medical diagnosis or treatment. Predictions can be wrong. Do not use this app to make healthcare decisions.")
st.caption("Numerical features only · No photo or scan analysis · Public Wisconsin breast cancer dataset")


@st.cache_resource(show_spinner="Training the reproducible baseline…")
def get_model():
    return train_model()


bundle = get_model()
try_tab, evidence_tab = st.tabs(["Try the model", "Model & evidence"])

with try_tab:
    st.subheader("Start with a public example")
    st.write("No measurements? Choose an example, then select **Run classification**. You can edit any value before running.")
    source = st.selectbox("Measurement source", [*EXAMPLES, "Enter my own measurements"])
    custom = source == "Enter my own measurements"
    defaults = {} if custom else example_values(source)
    if not custom:
        row_index = EXAMPLES[source]
        st.info(f"Dataset row {row_index} · Recorded label: {CLASS_NAMES[int(DATA.target[row_index])]} (not a new or independently validated case).")
    else:
        st.info("All 30 fields are required. Use only non-identifiable educational data, not real patient records.")
    st.caption("Inputs are processed on the server in memory. This app does not intentionally save submitted measurements. Hosting providers may retain operational logs; do not enter sensitive information.")
    st.write("**How to read the features**")
    st.caption("Mean = average across nuclei; error = standard error; worst = mean of the three largest values. Measurements must follow the dataset definitions, not everyday health measurements. Accepted ranges are the observed dataset ranges, not medical reference limits.")
    with st.form("measurements"):
        values = {}
        for group, start in [("Mean measurements", 0), ("Standard errors", 10), ("Worst measurements", 20)]:
            with st.expander(group, expanded=(start == 0)):
                columns = st.columns(2)
                for index in range(start, start + 10):
                    name = FEATURE_NAMES[index]
                    with columns[(index - start) % 2]:
                        values[name] = st.number_input(
                            name.capitalize(),
                            value=defaults.get(name),
                            format="%.6f",
                            step=0.000001,
                            key=f"{source}:{name}",
                            help=f"Dataset feature: {name}. Demo range: {LOWER_BOUNDS[index]:g}–{UPPER_BOUNDS[index]:g}. Keep the dataset's units and definitions.",
                        )
        submitted = st.form_submit_button("Run classification", type="primary")
    if submitted:
        try:
            predicted_class, label = predict_measurements(bundle, values)
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.subheader(f"Model prediction: {label}")
            st.write(f"Dataset class **{predicted_class}** · 0 = malignant; 1 = benign.")
            st.caption("This is a model class label, not a diagnosis or a calibrated estimate of cancer risk. A benign prediction does not establish that someone is healthy.")
    st.caption("A result reflects the last submitted values. After editing, select Run classification again to update it.")

with evidence_tab:
    st.subheader("An honest baseline")
    st.write("GaussianNB trained at startup on the bundled scikit-learn dataset. No external model downloads or pickle files are needed.")
    st.write(f"**{bundle.train_size} training rows · {bundle.test_size} held-out test rows · 30 features**")
    st.table(pd.DataFrame({
        "Held-out metric": ["Accuracy", "Malignant precision", "Malignant recall"],
        "Value": [f"{bundle.accuracy:.2%}", f"{bundle.malignant_precision:.2%}", f"{bundle.malignant_recall:.2%}"],
    }))
    st.caption("Precision: among malignant predictions, the fraction actually labeled malignant. Recall: among malignant test cases, the fraction detected by the model.")
    st.write("**Confusion matrix — counts, not percentages**")
    st.dataframe(pd.DataFrame(bundle.confusion, index=["Actual malignant (0)", "Actual benign (1)"], columns=["Predicted malignant (0)", "Predicted benign (1)"]), width="stretch")
    st.code("train_test_split(data, target, test_size=0.33, random_state=42)\nGaussianNB()", language="python")
    st.caption("Original algorithm and unstratified split preserved. These metrics use actual predictions on the held-out split; they do not use the unrelated five-label accuracy calculation in the original script.")
    st.subheader("Where the data comes from")
    st.write("The Wisconsin Diagnostic Breast Cancer dataset has 569 samples, 212 malignant and 357 benign. Features were computed from digitized images of fine needle aspirates of breast masses. This app accepts those extracted numerical features; it does not extract them from images.")
    st.markdown("[scikit-learn dataset documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html) · [UCI dataset and attribution](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)")
    st.caption("Source: Wolberg, Mangasarian, Street & Street (1993), UCI Machine Learning Repository. DOI: 10.24432/C5DW2B. Dataset license: CC BY 4.0.")
    st.subheader("What these results cannot show")
    st.write("A small historical dataset and a single split do not establish clinical reliability, generalization to other populations or instruments, or suitability for screening. The features are correlated, while GaussianNB assumes conditional independence. Public examples may belong to training or test data; running them is a demonstration, not independent validation. No clinical or fairness validation has been performed.")
    st.caption("Demo input bounds come from all 569 public rows solely to constrain this interface; they are not fitted preprocessing or clinical thresholds.")

st.divider()
st.caption("Built for learning and portfolio demonstration. No account, API key, or identifiable patient information is needed.")
