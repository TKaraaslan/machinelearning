
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="IPA Language Detector",
    page_icon="🌍",
    layout="wide"
)

st.title("IPA Language Detector")

st.info(
    "Identify languages using IPA (International Phonetic Alphabet) "
    "transcriptions with Machine Learning."
)

@st.cache_data
def load_data():
    url = "hf://datasets/omneity-labs/ipa-dict/data/train-00000-of-00001.parquet"
    return pd.read_parquet(url)

df = load_data()

st.success("Dataset loaded successfully!")

st.subheader("Dataset Information")

col1, col2 = st.columns(2)

with col1:
    st.metric("Total Records", f"{len(df):,}")

with col2:
    st.metric("Total Columns", len(df.columns))

st.subheader("Dataset Preview")

st.write("Column Names:", df.columns.tolist())

st.dataframe(
    df.head(20),
    use_container_width=True
)
