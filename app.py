from langdetect import detect
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, pipeline,
    MT5Tokenizer, BartTokenizer
)
import streamlit as st
import io

st.title("🔄 تلخيص ذكي متعدد اللغات")

user_input = st.text_area("أدخل النص هنا:", height=300)

if st.button("تلخيص"):
    if user_input.strip():
        with st.spinner("جاري التلخيص..."):
            lang = detect(user_input)

            if lang == 'ar':
                model_name = "csebuetnlp/mT5_multilingual_XLSum"
                tokenizer = MT5Tokenizer.from_pretrained(model_name, cache_dir="./model_cache")
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir="./model_cache")
            else:
                model_name = "facebook/bart-large-cnn"
                tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./model_cache")
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir="./model_cache")

            summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=-1)
            summary = summarizer(user_input, max_length=120, min_length=30, do_sample=False)
            summary_text = summary[0]["summary_text"]

            st.success("✅ الملخص:")
            st.write(summary_text)
            st.download_button("تحميل الملخص", data=summary_text, file_name="summary.txt")
    else:
        st.warning("الرجاء إدخال نص أولاً.")
