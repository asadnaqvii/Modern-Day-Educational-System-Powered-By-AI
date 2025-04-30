import os
import sys
import re
import io
import streamlit as st
from groq import Groq
from dotenv import load_dotenv

# ────────────────────────────────
# 0. ENV PATCHES (silence torch / file‑watcher errors)
# ────────────────────────────────
os.environ.setdefault("STREAMLIT_FILE_WATCHER_TYPE", "none")
import types
sys.modules.setdefault("torch.classes", types.ModuleType("torch.classes"))

# ────────────────────────────────
# 1. CONFIG & INIT
# ────────────────────────────────
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
SCOUT_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

def generate_explanation(concept, region, age_group, class_level, style_choice):
    """
    Stream from Groq’s LLaMA-4 Scout model.
    Yields partial explanation text as it arrives.
    """
    system_prompt = (
        "You are an expert educator. "
        "First think through your approach, then after the marker [START] output "
        "a clear, tailored explanation."
    )
    user_prompt = (
        f"### CONTEXT\n"
        f"- Concept: {concept}\n"
        f"- Region: {region}\n"
        f"- Age Group: {age_group}\n"
        f"- Class Level: {class_level}\n"
        f"- Style: {style_choice}\n\n"
        "[START]\n"
    )

    stream = client.chat.completions.create(
        model=SCOUT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.5,
        max_tokens=512,
        stream=True,
    )

    partial = ""
    for chunk in stream:
        # delta.content holds the next token (or None)
        token = chunk.choices[0].delta.content or ""
        partial += token
        yield partial


def main():
    st.set_page_config(layout="wide")
    st.markdown('<h1 style="text-align:center;">LLM Customized Explanation Generator</h1>', unsafe_allow_html=True)

    # sidebar params
    region = st.sidebar.selectbox("Region", ["North America","Europe","Asia","Africa","South America","Other"])
    age_group = st.sidebar.selectbox("Age Group", ["Under 10","10-12","13-15","16-18","Adult"])
    class_level = st.sidebar.selectbox("Class Level", ["Basic","Intermediate","Advanced"])
    style_choice = st.sidebar.selectbox("Style", ["Formal","Informal","Friendly","Detailed","Concise"])

    st.markdown("## Enter the Concept")
    concept = st.text_area("", height=150)

    if st.button("Generate Explanation"):
        if not concept.strip():
            st.error("Please enter a concept.")
            return

        placeholder = st.empty()
        with st.spinner("Generating…"):
            for partial in generate_explanation(concept, region, age_group, class_level, style_choice):
                # replace newlines with <br> if you want HTML
                placeholder.markdown(partial)

if __name__ == "__main__":
    main()
