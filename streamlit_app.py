import streamlit as st
import pandas as pd

st.title('Machine Learning App')

st.info('This app builds up machine learning model!')

df = pd.read_parquet("hf://datasets/omneity-labs/ipa-dict/data/train-00000-of-00001.parquet")
