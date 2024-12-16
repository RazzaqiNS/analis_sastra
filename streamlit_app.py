import streamlit as st
import spacy
import re
from collections import Counter
from PyPDF2 import PdfReader
from docx import Document
from googletrans import Translator
import nltk
import os
import io
import pandas as pd

# Ensure required nltk data and spaCy models are downloaded
nltk_data_dir = os.path.expanduser("~/nltk_data")
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)
nltk.data.path.append(nltk_data_dir)
nltk.download('punkt', download_dir=nltk_data_dir)
nltk.download('stopwords', download_dir=nltk_data_dir)

try:
    nlp = spacy.load("de_core_news_sm")
except OSError:
    from spacy.cli import download
    download("de_core_news_sm")
    nlp = spacy.load("de_core_news_sm")

# Function to read files
def read_file(filepath):
    ext = os.path.splitext(filepath.name)[1].lower()
    if ext == '.txt':
        return filepath.read().decode('utf-8')
    elif ext == '.pdf':
        reader = PdfReader(filepath)
        text = ''
        for page in reader.pages:
            text += page.extract_text() + '\n'
        return text
    elif ext == '.docx':
        doc = Document(filepath)
        text = ''
        for para in doc.paragraphs:
            text += para.text + '\n'
        return text
    else:
        st.error("Format file tidak didukung. Gunakan .txt, .pdf, atau .docx")
        return None

# Preprocess text
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text.lower()

# Analyze word frequency
def analyze_frequency(text):
    words = text.split()
    return Counter(words)

# Analyze POS
def analyze_pos(text):
    doc = nlp(text)
    return Counter([token.pos_ for token in doc])

# Expand words by POS
def expand_words_by_pos(text, pos_tags):
    doc = nlp(text)
    pos_words = {pos: [token.text for token in doc if token.pos_ == pos] for pos in pos_tags}
    return pos_words

# Translate text
translator = Translator()
def translate_text(text, src_lang, dest_lang):
    try:
        translated = translator.translate(text, src=src_lang, dest=dest_lang)
        return translated.text
    except Exception as e:
        return f"Error: {e}"

# Streamlit App
st.title("Analisis dan Terjemahan Teks Bahasa Jerman")

# File Upload
uploaded_file = st.file_uploader("Unggah file Anda (.txt, .pdf, .docx):", type=["txt", "pdf", "docx"])

if uploaded_file is not None:
    # Read file
    text = read_file(uploaded_file)
    if text:
        st.subheader("Teks Asli")
        st.text_area("Konten File:", text, height=200)

        # Preprocess text
        cleaned_text = preprocess_text(text)
        st.subheader("Teks Setelah Preprocessing")
        st.text_area("Konten Setelah Preprocessing:", cleaned_text, height=200)

        # Word Frequency Analysis
        st.subheader("Analisis Frekuensi Kata")
        word_freq = analyze_frequency(cleaned_text)
        df_freq = pd.DataFrame(word_freq.items(), columns=["Word", "Frequency"]).sort_values(by="Frequency", ascending=False)
        st.dataframe(df_freq.head(10))

        # POS Analysis
        st.subheader("Distribusi Kelas Kata")
        pos_counts = analyze_pos(cleaned_text)
        st.write(dict(pos_counts))

        # Words by POS
        st.subheader("Kata-kata Berdasarkan Kelas Kata")
        pos_tags = st.multiselect("Pilih Kelas Kata untuk Dijabarkan:", ['NOUN', 'VERB', 'ADJ'], default=['NOUN', 'VERB', 'ADJ'])
        expanded_words = expand_words_by_pos(cleaned_text, pos_tags)
        for pos, words in expanded_words.items():
            st.write(f"**{pos}:** {', '.join(words[:10])}")

        # Translation
        st.subheader("Terjemahan Teks")
        src_lang = st.selectbox("Pilih Bahasa Asal:", ["de", "id", "en"], index=0)
        dest_lang = st.selectbox("Pilih Bahasa Tujuan:", ["id", "en", "de"], index=1)
        if st.button("Terjemahkan Teks"):
            translated_text = translate_text(text, src_lang, dest_lang)
            st.text_area("Hasil Terjemahan:", translated_text, height=200)

        # Download Options
        st.subheader("Unduh Hasil Analisis")
        st.download_button(
            "Unduh Frekuensi Kata",
            data=df_freq.to_csv(index=False).encode('utf-8'),
            file_name="word_frequency.csv",
            mime="text/csv"
        )

        st.download_button(
            "Unduh Teks Terjemahan",
            data=translated_text.encode('utf-8'),
            file_name="translated_text.txt",
            mime="text/plain"
        )
