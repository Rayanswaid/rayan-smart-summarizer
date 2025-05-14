import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, MT5Tokenizer
from langdetect import detect
import io

st.set_page_config(page_title="Multilingual Text Summarizer", layout="centered")
st.title("🔄 Multilingual AI Text Summarizer")
st.write("Summarize English and Arabic text automatically using AI models.")

user_input = st.text_area("📝 Enter your Arabic or English text below:", height=300)

if st.button("🧠 Summarize"):
    if user_input.strip():
        with st.spinner("Summarizing..."):
            lang = detect(user_input)
            if lang == "ar":
                model_name = "csebuetnlp/mT5_multilingual_XLSum"
                tokenizer = MT5Tokenizer.from_pretrained(model_name, cache_dir="./model_cache")
            else:
                model_name = "facebook/bart-large-cnn"
                tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./model_cache")

            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir="./model_cache")
            summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=-1)
            summary = summarizer(user_input, max_length=120, min_length=30, do_sample=False)
            summary_text = summary[0]["summary_text"]
            st.success("✅ Summary:")
            st.write(summary_text)
            st.download_button("📥 Download Summary", data=summary_text, file_name="summary.txt")
    else:
        st.warning("⚠️ Please enter some text first.")

st.caption("🚀 Built with ❤️ by Rayan Swaid — 2025")
