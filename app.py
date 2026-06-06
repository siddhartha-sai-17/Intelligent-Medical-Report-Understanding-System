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
        background: linear-gradient(135deg, #eef2ff 0%, #f8fafc 100%);
        color: #0f172a;
    }
    .big-title {
        font-size: 36px;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.25rem;
    }
    .subtitle {
        color: #475569;
        font-size: 18px;
        margin-top: 0;
        margin-bottom: 1rem;
    }
    .card {
        background: rgba(255, 255, 255, 0.98);
        border-radius: 24px;
        padding: 24px;
        box-shadow: 0 18px 50px rgba(15, 23, 42, 0.08);
        border: 1px solid rgba(15, 23, 42, 0.08);
        margin-bottom: 24px;
    }
    .stButton>button {
        background-color: #0b5fff !important;
        color: white !important;
        border-radius: 12px !important;
        padding: 0.8rem 1.2rem !important;
        font-weight: 700 !important;
    }
    .stButton>button:hover {
        background-color: #2438ff !important;
    }
    .stTextArea textarea,
    .stTextInput>div>div>input {
        border-radius: 16px !important;
        background: #f8fafc !important;
        border: 1px solid rgba(15, 23, 42, 0.12) !important;
    }
    #MainMenu, footer, .viewerBadge_container__1QSob {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "report" not in st.session_state:
    st.session_state["report"] = ""

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
    ],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.header("Quick Examples")
example_reports = {
    "Cardiology (chest pain)": "Patient presents with chest pain, shortness of breath, and elevated troponin levels.",
    "Neurology (stroke)": "Sudden onset right-sided weakness, facial droop, and slurred speech noted 2 hours ago.",
    "Dermatology (rash)": "Widespread pruritic erythematous rash over trunk and extremities for 3 days."
}

for name, text in example_reports.items():
    if st.sidebar.button(name, key=f"example_{name}"):
        st.session_state["report"] = text

st.sidebar.markdown("---")
st.sidebar.info(
    "Upload a report, paste notes, or use an example. The analysis module predicts specialty and displays confidence scores."
)
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
    st.markdown('<div class="big-title">🩺 Medical Report Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Paste a clinical note or upload a report to classify the most likely specialty and review predicted confidence scores.</div>', unsafe_allow_html=True)

    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload report file (txt)", type=["txt"], help="Optional: upload a .txt clinical report")

        if uploaded is not None:
            st.session_state["report"] = uploaded.read().decode("utf-8")

        report = st.text_area(
            "Medical Report",
            value=st.session_state["report"],
            height=260,
            key="report_input"
        )

        st.markdown("---")
        analyze_btn = st.button("Analyze Report")
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Model quick info")
        st.markdown(f"**Max length:** {MAX_LENGTH}")
        st.markdown(f"**Vocabulary size:** {params.get('vocab_size', 'N/A'):,}")
        st.markdown(f"**Classes:** {params.get('num_classes', 'N/A')}")
        st.markdown("---")
        st.write("**Use one of the sidebar examples to populate the report quickly.**")
        st.markdown('</div>', unsafe_allow_html=True)

    if analyze_btn:
        report_text = st.session_state["report"]
        if not report_text or not report_text.strip():
            st.warning("Please enter or upload a medical report.")
        else:
            specialty, confidence, probs = predict_specialty(report_text)
            st.success(f"**Predicted Specialty:** {specialty}")

            metric_col1, metric_col2, metric_col3 = st.columns(3)
            metric_col1.metric("Confidence", f"{confidence * 100:.2f}%")
            metric_col2.metric("Vocabulary", f"{params.get('vocab_size', 'N/A'):,}")
            metric_col3.metric("Classes", params.get('num_classes', 'N/A'))

            class_names = label_encoder.classes_
            prob_df = pd.DataFrame({"Specialty": class_names, "Probability": probs})
            prob_df = prob_df.sort_values(by="Probability", ascending=False)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Top Predictions")
            topn = prob_df.head(5).reset_index(drop=True)
            cols = st.columns(len(topn))
            for i, row in topn.iterrows():
                with cols[i]:
                    st.markdown(f"**{i+1}. {row['Specialty']}**")
                    st.write(f"{row['Probability'] * 100:.1f}%")
                    st.progress(int(row['Probability'] * 100))

            st.subheader("Full Probability Distribution")
            chart = alt.Chart(prob_df).mark_bar().encode(
                x=alt.X('Probability:Q', title='Probability'),
                y=alt.Y('Specialty:N', sort='-x', title='Specialty'),
                tooltip=['Specialty', alt.Tooltip('Probability:Q', format='.3f')]
            ).properties(height=320)

            st.altair_chart(chart, use_container_width=True)
            with st.expander("View probability table"):
                st.dataframe(prob_df.style.format({"Probability": "{:.3f}"}))
            st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# PAGE 2
# =====================================================

elif page == "Medical Vocabulary Builder":

    st.markdown('<div class="big-title">📚 Medical Vocabulary Builder</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Explore the tokenizer vocabulary and search for medical terms recognized by the model.</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write(f"**Vocabulary Size:** {len(tokenizer.word_index):,}")

    search_word = st.text_input(
        "Search Medical Term"
    )

    if search_word:
        if search_word.lower() in tokenizer.word_index:
            st.success(f"'{search_word}' found.")
            st.write("Token Index:", tokenizer.word_index[search_word.lower()])
        else:
            st.error(f"'{search_word}' not found.")

    st.markdown("---")
    st.subheader("Top 100 Tokens")
    vocab_df = pd.DataFrame(
        list(tokenizer.word_index.items())[:100],
        columns=["Word", "Token_ID"]
    )
    st.dataframe(vocab_df)
    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# PAGE 3
# =====================================================

elif page == "Diagnostic Importance Analysis":

    st.markdown('<div class="big-title">🔎 Diagnostic Importance Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Highlight medically important terms in a report and inspect the most frequent known tokens.</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    report = st.text_area(
        "Enter Medical Report",
        value=st.session_state["report"],
        height=260
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

    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# PAGE 4
# =====================================================

elif page == "Model Information":

    st.markdown('<div class="big-title">📈 Model Information</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Inspect model configuration, supported specialties, and tokenizer information.</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.write(
        "**Maximum Sequence Length:**",
        params["max_length"]
    )

    st.write(
        "**Vocabulary Size:**",
        params["vocab_size"]
    )

    st.write(
        "**Number of Classes:**",
        params["num_classes"]
    )

    st.markdown("---")
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
    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")
st.caption(
    "Healthcare NLP | Self-Attention Medical Report Classification System"
)