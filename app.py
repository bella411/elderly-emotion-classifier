import streamlit as st
import joblib
import re
from pathlib import Path

@st.cache_resource
def load_model():
    model_path = Path(__file__).resolve().parent / "model/emotion_classifier.pkl"
    return joblib.load(model_path)

model = load_model()

def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()

st.title("노년층 발화 감정 분류")
st.write("문장을 입력하면 AI가 감정을 예측합니다.")

text = st.text_area("문장을 입력하세요.")

if st.button("감정 예측"):
    text = clean_text(text)

    if not text:
        st.warning("문장을 입력해주세요.")
    else:
        prediction = model.predict([text])[0]
        st.success(f"예측 감정: {prediction}")