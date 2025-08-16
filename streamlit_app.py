import os, io, base64, zipfile, time
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from PIL import Image
import streamlit as st

# -------- Claude (Anthropic) for vision + JSON --------
# pip install anthropic
from anthropic import Anthropic

# -------- Fast, free local embeddings (CPU) -----------
# pip install fastembed onnxruntime
from fastembed import TextEmbedding

ALLOWED_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

st.set_page_config(page_title="Screenshot Search (Claude + Free Local Embeddings)", page_icon="🖼️", layout="wide")
st.title("📸 Screenshot Search (Text + Visual)")
st.caption("Vision/understanding via Claude API. Embeddings are free & local (FastEmbed ONNX). Deployable on Streamlit, no GPU required.")

# ------------ API + model setup -------------
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    st.warning("Set ANTHROPIC_API_KEY in your environment / Streamlit Secrets.")

anthropic = Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

# Session state
if "index" not in st.session_state:
    st.session_state.index = []     # [{filename, thumbnail, combined_text, embedding, ...}]
if "matrix" not in st.session_state:
    st.session_state.matrix = None  # np.ndarray (N, D)
if "embedder_name" not in st.session_state:
    st.session_state.embedder_name = "sentence-transformers/all-MiniLM-L6-v2"
if "embedder" not in st.session_state:
    st.session_state.embedder = None

# ------------ Helpers -------------
def to_base64(img_bytes: bytes) -> str:
    return base64.b64encode(img_bytes).decode("utf-8")

def pillow_thumbnail(img_bytes: bytes, max_size: int = 512) -> Image.Image:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img.thumbnail((max_size, max_size))
    return img

def ensure_embedder(model_name: str):
    # Lazy-initialize a FastEmbed embedder; keeps memory small on cold start
    if st.session_state.embedder is None or st.session_state.embedder_name != model_name:
        st.session_state.embedder = TextEmbedding(model_name=model_name, cache_dir=".cache_fastembed")
        st.session_state.embedder_name = model_name
    return st.session_state.embedder

def embed_texts(texts: List[str], model_name: str, batch_size: int = 32) -> np.ndarray:
    """FastEmbed returns generator of vectors (L2 normalized by default)."""
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    embedder = ensure_embedder(model_name)
    vecs = []
    for batch_start in range(0, len(texts), batch_size):
        batch = texts[batch_start: batch_start + batch_size]
        for v in embedder.embed(batch):  # each v is a list[float]
            vecs.append(v)
    return np.array(vecs, dtype=np.float32)

def combine_fields(analysis: Dict[str, Any]) -> str:
    return "\n".join([
        analysis.get("ocr_text", ""),
        analysis.get("caption", ""),
        " ".join(analysis.get("ui_elements", []) or []),
        " ".join(analysis.get("colors", []) or []),
        " ".join(analysis.get("tags", []) or []),
    ]).strip()

def analyze_with_claude(img_bytes: bytes, model_name: str) -> Dict[str, Any]:
    """Vision+JSON with Claude via tool (strict schema)."""
    if anthropic is None:
        return {"ocr_text": "", "caption": "", "ui_elements": [], "colors": [], "tags": []}

    tool_name = "structured_screenshot_json"
    tools = [{
        "name": tool_name,
        "description": "Return strictly the requested JSON fields for a screenshot.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ocr_text": {"type": "string"},
                "caption": {"type": "string"},
                "ui_elements": {"type": "array", "items": {"type": "string"}},
                "colors": {"type": "array", "items": {"type": "string"}},
                "tags": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["ocr_text", "caption", "ui_elements", "colors", "tags"],
            "additionalProperties": False
        }
    }]

    prompt = (
        "You are a meticulous screenshot analyzer.\n"
        "1) Extract ALL visible on-screen text verbatim (OCR-like).\n"
        "2) Describe the UI (buttons, dialogs, icons, charts, colors, layout cues).\n"
        "3) Provide concise tags for search (e.g., 'error', 'auth', 'blue button', 'toast', 'dialog', 'settings').\n"
        "Return ONLY the tool call with JSON fields."
    )

    img_b64 = to_base64(img_bytes)
    msg = anthropic.messages.create(
        model=model_name,
        max_tokens=1024,
        tools=tools,
        tool_choice={"type": "tool", "name": tool_name},  # force JSON
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_b64}},
                {"type": "text", "text": prompt}
            ]
        }]
    )

    tool_input = {}
    for block in msg.content:
        if block.type == "tool_use" and block.name == tool_name:
            tool_input = block.input or {}
            break

    return {
        "ocr_text": (tool_input.get("ocr_text") or "").strip(),
        "caption": (tool_input.get("caption") or "").strip(),
        "ui_elements": tool_input.get("ui_elements") or [],
        "colors": tool_input.get("colors") or [],
        "tags": tool_input.get("tags") or []
    }

def add_records(records: List[Dict[str, Any]]):
    st.session_state.index.extend(records)
    if st.session_state.index:
        vecs = [r["embedding"] for r in st.session_state.index]
        st.session_state.matrix = np.vstack(vecs).astype(np.float32)
    else:
        st.session_state.matrix = None

# ------------ Sidebar -------------
with st.sidebar:
    st.subheader("Settings")
    claude_model = st.selectbox(
        "Claude vision model",
        ["claude-3-7-sonnet-latest", "claude-3-5-sonnet-latest", "claude-3-haiku-20240307"],
        index=0
    )
    embed_model = st.selectbox(
        "Local embedding model (FastEmbed)",
        [
            "sentence-transformers/all-MiniLM-L6-v2",   # 384-d, tiny & fast
            "BAAI/bge-small-en-v1.5"                    # 384-d, strong small model
        ],
        index=0
    )
    pause_s = st.number_input("Pause between API calls (sec)", 0.0, 2.0, 0.15, 0.05)
    st.caption("all-MiniLM is very light; bge-small-en gives a recall bump with similar footprint.")

    st.divider()
    st.markdown("### Upload")
    multi_files = st.file_uploader("Upload screenshots (multiple)", type=list(ALLOWED_EXTS), accept_multiple_files=True)
    zip_file = st.file_uploader("…or upload a ZIP", type=["zip"])

# ------------ Gather uploads -------------
uploads = []
if multi_files:
    for f in multi_files:
        if Path(f.name).suffix.lower() in ALLOWED_EXTS:
            uploads.append({"name": f.name, "bytes": f.read()})

if zip_file:
    with zipfile.ZipFile(zip_file) as zf:
        for zi in zf.infolist():
            p = Path(zi.filename)
            if not p.is_dir() and p.suffix.lower() in ALLOWED_EXTS:
                with zf.open(zi, "r") as fh:
                    uploads.append({"name": p.name, "bytes": fh.read()})

# ------------ Ingest & Search UI -------------
col_ingest, col_search = st.columns([1, 2], gap="large")

with col_ingest:
    st.subheader("1) Ingest screenshots")
    if st.button("Build / Extend Index", type="primary", disabled=(len(uploads) == 0 or anthropic is None)):
        with st.spinner(f"Analyzing {len(uploads)} image(s) with Claude + local embeddings…"):
            records = []
            for i, f in enumerate(uploads, 1):
                filename, img_bytes = f["name"], f["bytes"]
                analysis = analyze_with_claude(img_bytes, claude_model)
                combined = combine_fields(analysis)
                doc_vec = embed_texts([combined], model_name=embed_model)[0]
                thumb = pillow_thumbnail(img_bytes, max_size=480)
                records.append({
                    "filename": filename,
                    "thumbnail": thumb,
                    "ocr_text": analysis["ocr_text"],
                    "caption": analysis["caption"],
                    "ui_elements": analysis["ui_elements"],
                    "colors": analysis["colors"],
                    "tags": analysis["tags"],
                    "combined_text": combined,
                    "embedding": doc_vec
                })
                time.sleep(pause_s)
            add_records(records)
        st.success(f"Indexed {len(records)} screenshot(s).")
    st.markdown(f"**Indexed images:** {len(st.session_state.index)}")

with col_search:
    st.subheader("2) Search")
    q = st.text_input('Query (e.g., "error message about auth", "screenshot with blue button")')
    if st.button("Search", type="primary", disabled=(not q or st.session_state.matrix is None)):
        with st.spinner("Searching…"):
            q_vec = embed_texts([q], model_name=embed_model)[0]  # FastEmbed returns L2-normalized vectors
            M = st.session_state.matrix  # (N, D), also normalized
            sims = (M @ q_vec)  # dot == cosine for normalized vectors
            order = np.argsort(-sims)[:5].tolist()

            st.markdown("### Top 5 matches")
            for rank, idx in enumerate(order, 1):
                rec = st.session_state.index[idx]
                sim = float(sims[idx])
                conf = round((sim + 1.0) / 2.0, 3)  # map [-1,1] -> [0,1]
                with st.container(border=True):
                    c1, c2 = st.columns([1, 2], vertical_alignment="top")
                    with c1:
                        st.image(rec["thumbnail"], caption=rec["filename"], use_container_width=True)
                        st.metric("Confidence", f"{conf:.3f}")
                    with c2:
                        st.markdown(f"**Caption:** {rec['caption'] or '—'}")
                        if rec["tags"]:
                            st.markdown("**Tags:** " + ", ".join(rec["tags"]))
                        if rec["ui_elements"]:
                            st.markdown("**UI elements:** " + ", ".join(rec["ui_elements"]))
                        if rec["colors"]:
                            st.markdown("**Colors:** " + ", ".join(rec["colors"]))
                        if rec["ocr_text"]:
                            with st.expander("OCR text"):
                                st.write(rec["ocr_text"][:4000])

    st.caption("Confidence = normalized similarity derived from unit-length embeddings (≈ cosine).")
