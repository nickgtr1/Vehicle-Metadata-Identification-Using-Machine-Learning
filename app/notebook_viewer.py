"""Render .ipynb files inline in Streamlit, for a 'Notebooks' tab showing the
VLM work (zero-shot eval, fine-tuning comparison, etc.) without leaving the app.

Uses nbconvert's "basic" template (bare content, no built-in stylesheet) wrapped
in minimal dark-theme CSS matching the rest of the app, rather than the "lab"/
"classic" templates -- those ship a full Jupyter stylesheet that assumes a wide,
light-background page and renders cramped/illegible in a narrow or dark container.
"""
from pathlib import Path

import nbformat
import streamlit as st
import streamlit.components.v1 as components
from nbconvert import HTMLExporter

_CSS = """
<style>
  body { background:#0e1117; color:#e6e6e6; font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;
         font-size:14px; line-height:1.5; margin:0; padding:1rem; }
  pre, code { background:#1a1d24; color:#e6e6e6; border-radius:6px; }
  pre { padding:.75rem; overflow-x:auto; white-space:pre-wrap; word-break:break-word; }
  .jp-RenderedMarkdown h1, .jp-RenderedMarkdown h2, .jp-RenderedMarkdown h3 { color:#fff; }
  table { border-collapse:collapse; width:100%; margin:.5rem 0; }
  th, td { border:1px solid rgba(255,255,255,.15); padding:.3rem .6rem; text-align:left; }
  img, svg { max-width:100%; height:auto; }
  .jp-OutputArea-output { overflow-x:auto; }
</style>
"""


@st.cache_data(show_spinner=False)
def _convert_notebook_to_html(path_str: str, mtime: float) -> str:
    """mtime is part of the cache key so an edited-and-resaved notebook gets
    re-rendered instead of silently showing stale cached HTML."""
    nb = nbformat.read(path_str, as_version=4)
    exporter = HTMLExporter(
        template_name="basic",
        embed_images=True,
        exclude_input=True,          # hide code cells entirely
        exclude_input_prompt=True,   # drop the "In [3]:" markers
        exclude_output_prompt=True,  # drop the "Out[3]:" markers
    )
    body, _resources = exporter.from_notebook_node(nb)
    return f"<!DOCTYPE html><html><head>{_CSS}</head><body>{body}</body></html>"


def render_notebooks_tab(notebooks_dir: Path, height: int = 1400) -> None:
    """Call this unconditionally, with no st.stop() anywhere above it in the
    script, or it will never render once the rest of the app calls st.stop()
    (e.g. before an image is uploaded). Needs a full-width container (a main-area
    tab, not the sidebar) -- notebook content with dataframes/plots is too wide
    for a ~300px sidebar column."""
    notebooks = sorted(notebooks_dir.glob("*.ipynb")) if notebooks_dir.is_dir() else []
    if not notebooks:
        st.info(f"No notebooks found in {notebooks_dir}")
        return

    choice = st.selectbox("Notebook", notebooks, format_func=lambda p: p.name)
    try:
        html = _convert_notebook_to_html(str(choice), choice.stat().st_mtime)
    except Exception as exc:  # noqa: BLE001 -- show the error, don't crash the app
        st.error(f"Could not render {choice.name}: {type(exc).__name__}: {exc}")
        return
    components.html(html, height=height, scrolling=True)