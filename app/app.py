import streamlit as st
import whisper
import requests
import pandas as pd
import numpy as np
import os
from pathlib import Path

# ============================================================
# 🎨 CUSTOM STYLING
# ============================================================
st.set_page_config(page_title="Arabic Audio Intelligence Pro", layout="wide", page_icon="🎙️")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .arabic-text { font-family: 'Amiri', serif; direction: rtl; text-align: right; font-size: 20px; line-height: 1.6; padding: 15px; background: #161b22; border-radius: 8px; border-right: 5px solid #1f6feb; }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# ⚙️ MODEL LOADING
# ============================================================
@st.cache_resource
def load_asr_model(size):
    return whisper.load_model(size)

# ============================================================
# 🧠 AI LOGIC (OLLAMA - THE SMART WAY)
# ============================================================
def summarize_with_ollama(text, model_name):
    prompt = f"أنت خبير في تلخيص النصوص العربية. قم بتلخيص النص التالي في فقرة واحدة موجزة وواضحة باللغة العربية فقط:\n\nالنص: {text}\n\nالملخص:"
    try:
        response = requests.post("http://localhost:11434/api/generate", json={
            "model": model_name,
            "prompt": prompt,
            "stream": False
        }, timeout=60)
        data = response.json()
        
        # Support both 'response' and 'message' formats
        if 'response' in data:
            return data['response'].strip()
        elif 'message' in data:
            return data['message']['content'].strip()
        else:
            return f"AI Output: {data}"
    except Exception as e:
        return f"⚠️ Error: {e}. Is Ollama running?"

# ============================================================
# 🖥️ UI LAYOUT
# ============================================================
st.sidebar.title("🛠️ System Configuration")
whisper_size = st.sidebar.selectbox("Whisper Size", ["tiny", "base", "small", "medium"], index=3)
ollama_model = st.sidebar.text_input("Ollama Model", value="qwen:0.5b")

st.title("🎙️ Arabic Audio Intelligence Pro")
st.markdown("---")

with st.spinner("🔄 Initializing Whisper ASR..."):
    asr_model = load_asr_model(whisper_size)

tab1, tab2 = st.tabs(["🚀 Process Audio", "📊 Analytics"])

with tab1:
    uploaded_file = st.file_uploader("Upload Arabic .wav", type=["wav"])
    if uploaded_file:
        st.audio(uploaded_file)
        if st.button("Start Pipeline"):
            with st.spinner("Step 1: Transcribing..."):
                with open("temp.wav", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                transcript = asr_model.transcribe("temp.wav", language='ar')['text']
                st.session_state['t'] = transcript
            
            with st.spinner(f"Step 2: Summarizing with {ollama_model}..."):
                summary = summarize_with_ollama(transcript, ollama_model)
                st.session_state['s'] = summary
            
            st.success("Complete!")

    if 't' in st.session_state:
        st.markdown("**📄 Full Transcript**")
        st.markdown(f'<div class="arabic-text">{st.session_state["t"]}</div>', unsafe_allow_html=True)
        st.markdown("**✨ AI Summary**")
        st.markdown(f'<div class="arabic-text">{st.session_state["s"]}</div>', unsafe_allow_html=True)

st.sidebar.info(f"💾 Models cached in: {Path.home()}/.cache")
