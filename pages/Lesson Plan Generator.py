import os
import math
from dotenv import load_dotenv

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from groq import Groq

# ─────────────────────────────────────────────────────────────
# 1. CONFIG & INIT
# ─────────────────────────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    st.error("GROQ_API_KEY not found. Add it to your .env")
    st.stop()

client = Groq(api_key=API_KEY)
MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

st.set_page_config(page_title="AI Lesson Plan Generator", layout="wide")
st.title("🗓️ AI‑Powered Weekly Lesson Plan Generator")

# ─────────────────────────────────────────────────────────────
# 2. SIDEBAR INPUTS
# ─────────────────────────────────────────────────────────────
st.sidebar.header("Upload & Settings")
curriculum_pdf = st.sidebar.file_uploader("📥 Upload Full Curriculum (PDF)", type=["pdf"])
total_days = st.sidebar.number_input(
    "Total working days to cover", min_value=1, value=200, step=1,
    help="Enter the total number of teaching days you have for this course."
)
days_per_week = st.sidebar.number_input(
    "Working days per week", min_value=1, max_value=7, value=5,
    help="How many days per week do you teach?"
)

# ─────────────────────────────────────────────────────────────
# 3. EXTRACT & PREP TEXT
# ─────────────────────────────────────────────────────────────
if not curriculum_pdf:
    st.info("Please upload your curriculum PDF to proceed.")
    st.stop()

with st.spinner("📖 Extracting text from PDF..."):
    reader = PdfReader(curriculum_pdf)
    pages = [page.extract_text() or "" for page in reader.pages]
    full_text = "\n".join(pages)

if not full_text.strip():
    st.error("No text could be extracted. Is your PDF text‑based?")
    st.stop()

st.success("✅ Curriculum text extracted.")

# ─────────────────────────────────────────────────────────────
# 4. SPLIT INTO DAY‑CHUNKS
# ─────────────────────────────────────────────────────────────
# Calculate chunk size so we get exactly `total_days` segments
total_chars = len(full_text)
chunk_size = math.ceil(total_chars / total_days)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=0,
    separators=["\n"]
)
day_chunks = splitter.split_text(full_text)

# Trim any excess chunks
if len(day_chunks) > total_days:
    day_chunks = day_chunks[:total_days]

# Compute how many weeks we need
num_weeks = math.ceil(len(day_chunks) / days_per_week)

# ─────────────────────────────────────────────────────────────
# 5. GENERATE & DISPLAY LESSON PLAN
# ─────────────────────────────────────────────────────────────
st.header("Generated Weekly Lesson Plan")

for week in range(1, num_weeks + 1):
    start_idx = (week - 1) * days_per_week
    week_chunks = day_chunks[start_idx : start_idx + days_per_week]

    # Build LLM prompt
    prompt = (
        f"Create a detailed lesson plan for **Week {week}** "
        f"with **{len(week_chunks)}** teaching days (out of {days_per_week} per week). "
        "Use the following curriculum excerpts for each day:\n\n"
        + "\n---\n".join(
            f"**Day {i+1}:** {seg}"
            for i, seg in enumerate(week_chunks)
        )
    )

    with st.spinner(f"🤖 Generating plan for Week {week}…"):
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are an expert educator."},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.5,
            max_tokens=1024,
            stream=False,
        )
    plan = resp.choices[0].message.content.strip()

    st.subheader(f"Week {week}")
    st.markdown(plan)

# ─────────────────────────────────────────────────────────────
# 6. FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("*Powered by Groq LLaMA‑4 Scout & Streamlit*")
