import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
import altair as alt

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Intelligent Medical Report Understanding System",
    page_icon="🩺",
    layout="wide"
)

# Custom styling
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(90deg, #f8fafc 0%, #ffffff 100%);
        color: #0f172a;
    }
    .big-title {
        font-size:32px;
        font-weight:700;
        color:#0b5fff;
    }
    .subtitle {
        color:#475569;
        margin-bottom:12px;
    }
    .stButton>button {
        background-color: #0b5fff;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# =====================================================
# LOAD FILES
# =====================================================

@st.cache_resource
def load_resources():

    model = tf.keras.models.load_model(
        "medical_model.keras",
        compile=False
    )

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    with open("model_params.pkl", "rb") as f:
        params = pickle.load(f)

    return model, tokenizer, label_encoder, params


model, tokenizer, label_encoder, params = load_resources()

MAX_LENGTH = params["max_length"]

# =====================================================
# HEADER
# =====================================================

header_col, _, help_col = st.columns([6, 1, 3])

with header_col:
    st.markdown('<div class="big-title">🩺 Intelligent Medical Report Understanding System</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Analyze medical reports, surface likely specialties, and explore important terms.</div>', unsafe_allow_html=True)

with help_col:
    st.info("Upload or paste a report, then click **Analyze Report**. Use the sidebar to switch modules.")

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Module",
    [
        "Medical Report Analysis",
        "Medical Vocabulary Builder",
        "Diagnostic Importance Analysis",
        "Model Information"
    ]
)

st.sidebar.markdown("---")
st.sidebar.header("Quick Examples")
st.sidebar.write("Try pasting one of these example reports:")
example_reports = {
    "Cardiology (chest pain)": "Patient presents with chest pain, shortness of breath, and elevated troponin levels.",
    "Neurology (stroke)": "Sudden onset right-sided weakness, facial droop, and slurred speech noted 2 hours ago.",
    "Dermatology (rash)": "Widespread pruritic erythematous rash over trunk and extremities for 3 days."
}

for name, text in example_reports.items():
    st.sidebar.button(name, key=f"example_{name}")

st.sidebar.markdown("---")
st.sidebar.caption("Built for clinicians and researchers — handle PHI appropriately.")

# =====================================================
# PREPROCESSING
# =====================================================

def preprocess_text(text):

    sequence = tokenizer.texts_to_sequences([text])

    padded = pad_sequences(
        sequence,
        maxlen=MAX_LENGTH,
        padding="post",
        truncating="post"
    )

    return padded


# =====================================================
# PREDICTION
# =====================================================

def predict_specialty(text):

    processed_text = preprocess_text(text)

    prediction = model.predict(
        processed_text,
        verbose=0
    )

    predicted_class = np.argmax(prediction)

    specialty = label_encoder.inverse_transform(
        [predicted_class]
    )[0]

    confidence = float(np.max(prediction))

    return specialty, confidence, prediction[0]


# =====================================================
# PAGE 1
# =====================================================

if page == "Medical Report Analysis":
    st.header("Medical Report Classification")

    left, right = st.columns([3, 2])

    with left:
        uploaded = st.file_uploader("Upload report file (txt) ", type=["txt"], help="Optional: upload a .txt clinical report")

        if uploaded is not None:
            report = uploaded.read().decode("utf-8")
            st.text_area("Report Preview", value=report, height=220)
        else:
            report = st.text_area("Paste Medical Report", height=220, placeholder="Paste clinical note or report here...")

        example_select = st.selectbox("Or use an example report", ["","Cardiology (chest pain)","Neurology (stroke)","Dermatology (rash)"])
        if example_select and report.strip()=="":
            report = example_reports.get(example_select.split(" (")[0]+" ("+example_select.split("(")[-1], report)

        analyze_btn = st.button("Analyze Report")

    with right:
        st.markdown("**Model quick info**")
        st.write(f"Max length: {MAX_LENGTH}")
        st.write(f"Vocab size: {params.get('vocab_size', 'N/A'):,}")
        st.write(f"Classes: {params.get('num_classes', 'N/A')}")

    if analyze_btn:
        if not report or report.strip() == "":
            st.warning("Please enter or upload a medical report.")
        else:
            specialty, confidence, probs = predict_specialty(report)

            st.success(f"Predicted Specialty: {specialty}")
            st.metric("Confidence", f"{confidence*100:.2f}%")

            class_names = label_encoder.classes_
            prob_df = pd.DataFrame({"Specialty": class_names, "Probability": probs})
            prob_df = prob_df.sort_values(by="Probability", ascending=False)

            st.subheader("Top Predictions")
            topn = prob_df.head(5).reset_index(drop=True)

            cols = st.columns(len(topn))
            for i, row in topn.iterrows():
                with cols[i]:
                    st.markdown(f"**{i+1}. {row['Specialty']}**")
                    st.write(f"{row['Probability']*100:.1f}%")
                    st.progress(int(row['Probability']*100))

            st.subheader("Full Probability Distribution")
            chart = alt.Chart(prob_df).mark_bar().encode(
                x=alt.X('Probability:Q'),
                y=alt.Y('Specialty:N', sort='-x'),
                tooltip=['Specialty', alt.Tooltip('Probability:Q', format='.3f')]
            ).properties(height=300)

            st.altair_chart(chart, use_container_width=True)

            with st.expander("View probability table"):
                st.dataframe(prob_df.style.format({"Probability": "{:.3f}"}))

# =====================================================
# PAGE 2
# =====================================================

elif page == "Medical Vocabulary Builder":

    st.header("Medical Vocabulary Builder")

    st.write(
        f"Vocabulary Size: {len(tokenizer.word_index):,}"
    )

    search_word = st.text_input(
        "Search Medical Term"
    )

    if search_word:

        if search_word.lower() in tokenizer.word_index:

            st.success(
                f"'{search_word}' found."
            )

            st.write(
                "Token Index:",
                tokenizer.word_index[
                    search_word.lower()
                ]
            )

        else:
            st.error(
                f"'{search_word}' not found."
            )

    st.subheader("Sample Vocabulary")

    vocab_df = pd.DataFrame(
        list(tokenizer.word_index.items())[:100],
        columns=["Word", "Token_ID"]
    )

    st.dataframe(vocab_df)

# =====================================================
# PAGE 3
# =====================================================

elif page == "Diagnostic Importance Analysis":

    st.header("Diagnostic Importance Analysis")

    report = st.text_area(
        "Enter Medical Report",
        height=250
    )

    if st.button("Analyze Important Terms"):

        words = report.lower().split()

        words = [
            word.strip(".,!?;:()[]{}")
            for word in words
        ]

        freq = {}

        for word in words:

            if word in tokenizer.word_index:

                freq[word] = freq.get(word, 0) + 1

        if len(freq) > 0:

            importance_df = pd.DataFrame(
                freq.items(),
                columns=["Medical Term", "Frequency"]
            )

            importance_df = importance_df.sort_values(
                by="Frequency",
                ascending=False
            )

            st.dataframe(importance_df)

            st.bar_chart(
                importance_df.set_index(
                    "Medical Term"
                )
            )

        else:

            st.warning(
                "No known medical terms found."
            )

# =====================================================
# PAGE 4
# =====================================================

elif page == "Model Information":

    st.header("Model Information")

    st.write(
        "Maximum Sequence Length:",
        params["max_length"]
    )

    st.write(
        "Vocabulary Size:",
        params["vocab_size"]
    )

    st.write(
        "Number of Classes:",
        params["num_classes"]
    )

    st.subheader("Supported Specialties")

    specialties = pd.DataFrame({
        "Specialties":
        label_encoder.classes_
    })

    st.dataframe(specialties)

    st.subheader("Model Summary")

    summary_lines = []

    model.summary(
        print_fn=lambda x:
        summary_lines.append(x)
    )

    st.text("\n".join(summary_lines))

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")
st.caption(
    "Healthcare NLP | Self-Attention Medical Report Classification System"
)