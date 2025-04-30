import streamlit as st

# Set up the main homepage configuration
st.set_page_config(
    page_title="Modern Day Educational System, Powered By AI",
    page_icon="📚",
    layout="wide"
)

# Header
st.markdown(
    "<h1 style='text-align: center;'>Modern Day Educational System, Powered By AI</h1>",
    unsafe_allow_html=True
)

# Introduction
st.write(
    "Welcome to the Modern Day Educational System powered by AI!"
)
st.write(
    "Use the sidebar menu (or the 'Pages' dropdown in Streamlit) to navigate through the following modules:"
)

# List of modules
st.markdown("""
- 🧾 **AI Exam Evaluator** — OCR + Vision LLM
- 📘 **AI Question Paper Generator** — OCR Edition
- 🗓️ **AI Timetable Generator** — Groq + LLaMA-4
- ✏️ **Explanation Generator** — LLM Customized Explanation Generator
- 📚 **Lesson Plan Generator** — Groq + LLaMA-4
"""
)
