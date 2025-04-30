import os
import sys
import re
import io
import base64
import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
from groq import Groq
from dotenv import load_dotenv

# ────────────────────────────────
# 0. ENV PATCHES
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

# ────────────────────────────────
# 2. INPUT HANDLING
# ────────────────────────────────
def render_page_to_image(page: fitz.Page) -> Image.Image:
    """Render a PDF page at 2× resolution to a PIL Image."""
    zoom = 2.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def extract_pages(file) -> list[Image.Image]:
    """Return list of PIL Images for each PDF page or a single image file."""
    if file is None:
        return []
    data = file.read()
    ctype = file.type
    if "pdf" in ctype:
        doc = fitz.open(stream=data, filetype="pdf")
        return [render_page_to_image(page) for page in doc]
    elif "image" in ctype:
        return [Image.open(io.BytesIO(data))]
    else:
        st.error("Unsupported file type. Please upload a PDF or an image.")
        return []


def encode_image_to_data_url(img: Image.Image) -> str:
    """Encode a PIL Image to a base64 data URL."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

# ────────────────────────────────
# 3. CHUNKING FUNCTION
# ────────────────────────────────
def chunk_pages(pages: list, size: int = 5) -> list:
    """Yield successive size-sized chunks from pages."""
    for i in range(0, len(pages), size):
        yield pages[i : i + size]

# ────────────────────────────────
# 4. STREAMLIT UI
# ────────────────────────────────
st.set_page_config("AI Question Paper Generator", layout="wide")
st.title("📘 AI Question Paper Generator — Vision LLM Edition")

with st.sidebar:
    st.header("📥 Upload Past Paper")
    paper_file = st.file_uploader(
        "Upload a PDF or Image of a past exam paper", type=["pdf", "png", "jpg", "jpeg"]
    )
    subject = st.text_input("🧠 Subject / Topic (optional)")
    temperature = st.slider("🎛️ LLM temperature", 0.0, 1.0, 0.7, 0.05)
    generate_btn = st.button("🚀 Generate New Paper")

# ────────────────────────────────
# 5. DISPLAY UPLOADED PAGES
# ────────────────────────────────
pages = extract_pages(paper_file)
if pages:
    st.subheader("📄 Uploaded Paper Preview")
    cols = st.columns(len(pages))
    for col, img in zip(cols, pages):
        col.image(img, use_container_width=True)

# ────────────────────────────────
# 6. GENERATION LOGIC
# ────────────────────────────────
def generate_question_paper(pages: list, subject: str, temp: float) -> str:
    """
    Uses the Vision LLM to analyze the uploaded paper images and generate
    a new question paper following the same structure and format.
    """
    content = []
    for idx, img in enumerate(pages, start=1):
        content.append({"type": "text", "text": f"--- Page {idx} ---"})
        content.append({"type": "image_url", "image_url": {"url": encode_image_to_data_url(img)}})

    subj_clause = (
        f"Subject / Topic: {subject}." if subject.strip()
        else "Infer the subject/topic from the paper."
    )

    system_prompt = (
        "You are an expert exam setter. The provided images contain a past exam paper "
        "with questions, sections, and mark allocations.\n"
        f"{subj_clause}\n\n"
        "STEP 1: Analyze the paper to identify the structure (sections, question types, mark allocations).\n"
        "STEP 2: Generate a brand-new paper that follows the same format exactly, creating original questions.\n"
        "Return only the new question paper, preserving headings, numbering, and mark scheme style."
    )

    response = client.chat.completions.create(
        model=SCOUT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": content},
        ],
        temperature=temp,
        max_tokens=2048,
        stream=False,
    )
    return response.choices[0].message.content.strip()

# ────────────────────────────────
# 7. RUN GENERATION WITH CHUNKING
# ────────────────────────────────
if generate_btn:
    if not pages:
        st.error("Please upload a past exam paper first.")
        st.stop()

    with st.spinner("🤖 Generating new question paper…"):
        try:
            generated_parts = []
            # Generate in chunks of up to 5 pages
            for chunk in chunk_pages(pages, size=5):
                part = generate_question_paper(chunk, subject, temperature)
                generated_parts.append(part)

            # Combine all parts into one final paper
            new_paper = "\n\n".join(generated_parts)

            st.subheader("📝 Generated Question Paper")
            st.code(new_paper)
            st.download_button(
                "📥 Download New Paper",
                new_paper,
                file_name=f"{(subject or 'new_paper').replace(' ', '_')}.txt",
                mime="text/plain"
            )
        except Exception as e:
            st.error(f"Generation failed: {e}")