"""
SOP Chatbot — Gradio Web Interface (V5).
Professional dark-themed UI for querying SOP knowledge base.

Usage:
    python app.py
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from core.pipeline import SOPPipeline
from core.rewriter import check_ollama_status

import gradio as gr


# ── Global Pipeline ──────────────────────────────────
pipeline = None


def initialize_pipeline():
    """Load the pipeline on startup."""
    global pipeline
    pipeline = SOPPipeline()

    # Check if index exists
    if os.path.exists(config.FAISS_INDEX_FILE) and os.path.exists(config.ID_MAP_FILE):
        pipeline.load()
    else:
        # Build from scratch
        pipeline.build()

    return pipeline


def answer_question(question: str, use_rewrite: bool = False) -> tuple:
    """
    Process a user question through the SOP pipeline.
    Returns formatted results for the Gradio UI.
    """
    global pipeline

    if not question or question.strip() == "":
        return (
            "",            # answer
            "",            # status
            "",            # matched question
            "",            # score
            "",            # debug info
        )

    # Temporarily set rewrite mode
    original_rewrite = config.USE_LLM_REWRITE
    config.USE_LLM_REWRITE = use_rewrite

    start = time.time()
    result = pipeline.query_detailed(question, top_k=3)
    elapsed = time.time() - start

    # Restore rewrite setting
    config.USE_LLM_REWRITE = original_rewrite

    # Format answer
    if result["rejected"]:
        answer = f"❌ {result['answer']}"
        status = f"🔴 REJECTED — Below threshold ({result['score']:.4f} < {result['threshold']:.3f})"
    else:
        answer = f"✅ {result['answer']}"
        status = f"🟢 MATCHED — Confidence: {result['score']:.4f} (threshold: {result['threshold']:.3f})"

    matched = result["matched_question"] or "No match found"
    score = f"{result['score']:.4f}" if result["score"] > 0 else "N/A"

    # Debug: top-K results
    debug_lines = [f"⏱️ Response time: {elapsed*1000:.1f}ms", ""]
    debug_lines.append("Top matches:")
    for i, r in enumerate(result.get("top_k_results", []), 1):
        marker = "→" if i == 1 else " "
        rerank = f" | R: {r.get('score', 0):.3f}"
        dense = f" | D: {r.get('dense_score', 0):.3f}"
        bm25 = f" | BM25: {r.get('bm25_score', 0):.3f}"
        debug_lines.append(
            f"  {marker} [{i}]{rerank}{dense}{bm25} | Q: {r['entry']['question'][:80]}"
        )
    debug_info = "\n".join(debug_lines)

    return answer, status, matched, score, debug_info


def get_system_status() -> str:
    """Get system status information."""
    status_lines = [
        "── V6 Pipeline (BGE-M3 Only) ──",
        f"📦 Embedding: {config.EMBEDDING_MODEL}",
        f"🔍 Reranking: Cross-Encoder (Enabled)",
        f"📊 Threshold: 0.60 / 0.25 (Dual)",
        f"📂 FAISS Entries: {pipeline.faiss_index.index.ntotal if pipeline and pipeline.faiss_index.index else 'N/A'}",
        # f"📖 BM25 Entries: {len(pipeline.bm25_index.entries) if pipeline and pipeline.bm25_index.entries else 'N/A'}",
        # f"🎯 Fusion: {'Trained' if pipeline and pipeline.fusion.is_trained else 'Default weights'}",
        # f"📐 Calibrator: {'Trained' if pipeline and pipeline.calibrator.is_trained else 'Default sigmoid'}",
    ]

    # Check Ollama
    ollama_ok = check_ollama_status()
    status_lines.append(f"🤖 Ollama ({config.OLLAMA_MODEL}): {'✅ Available' if ollama_ok else '❌ Not available'}")

    return "\n".join(status_lines)


# ── Custom CSS ───────────────────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* {
    font-family: 'Inter', sans-serif !important;
}

.gradio-container {
    max-width: 1100px !important;
    margin: auto !important;
}

.main-header {
    text-align: center;
    padding: 24px 0 12px 0;
}

.main-header h1 {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
}

.main-header p {
    color: #888;
    font-size: 0.95rem;
    margin: 0;
}

.answer-box {
    font-size: 1.15rem !important;
    line-height: 1.7 !important;
    padding: 16px !important;
    border-radius: 12px !important;
    min-height: 80px !important;
}

.status-box textarea {
    font-weight: 600 !important;
    font-size: 0.95rem !important;
}

.debug-box textarea {
    font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
    font-size: 0.82rem !important;
    line-height: 1.5 !important;
    color: #aaa !important;
}

footer {
    display: none !important;
}

.pill-badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
"""


# ── Build Gradio App ─────────────────────────────────
def format_status_html(result):
    """Format the status/match info as an HTML block."""
    if not result["answer"]:
        return "<div style='padding:10px; color:#666;'>Ready to search...</div>"

    if result["rejected"]:
        color = "#ef4444"
        bg = "#fee2e2"
        icon = "🔴"
        title = "REJECTED"
        desc = "Question appears out-of-scope or contradictory."
    else:
        color = "#10b981"
        bg = "#d1fae5"
        icon = "🟢"
        title = "MATCHED"
        desc = f"Confidence: <b>{result['score']:.4f}</b>"

    matched_q = result.get('matched_question', 'N/A')
    
    html = f"""
    <div style="border: 1px solid {color}; background-color: {bg}10; border-radius: 8px; padding: 12px;">
        <div style="color: {color}; font-weight: bold; font-size: 1.1em; margin-bottom: 4px;">
            {icon} {title} &nbsp;&nbsp;<span style="font-size:0.9em; font-weight:normal; color:#555;">{desc}</span>
        </div>
        <div style="margin-top: 8px; font-size: 0.95em; color: #444;">
            <span style="font-weight:600;">Matched SOP Question:</span><br>
            <span style="font-style:italic;">"{matched_q}"</span>
        </div>
    </div>
    """
    return html

def create_app():
    """Create and configure the Gradio app."""
    
    # Define example categories
    examples_color = [
        ["What is the PMS value of silver standard?"],
        ["Should black be set to 100 percent K for single color print?"],
        ["Should gradients be used in one-color screen print?"],
        ["What is the PMS value of gold standard?"],
    ]
    examples_artwork = [
        ["Should text be converted to outlines?"],
        ["What file formats are acceptable for vector artwork?"],
        ["Should transparency be flattened before submission?"],
        ["What should be done if artwork is blurry?"],
    ]
    examples_type = [
        ["What is the minimum font size for positive sans-serif text?"],
        ["Should fonts be embedded in final PDF?"],
        ["What is the minimum stroke thickness for negative space?"],
    ]
    examples_oos = [
        ["How do I cook pasta?"],
        ["What is the weather today?"],
        ["Tell me a joke."],
    ]

    with gr.Blocks(title="SOP Chatbot V6", css=CUSTOM_CSS) as app:

        # ── Header ──────────────────────────────
        gr.HTML("""
        <div class="main-header">
            <h1>🛡️ SOP Chatbot V6</h1>
            <p>BGE-M3 (1024d) • GPU Accelerated • Zero Hallucination Guardrails</p>
        </div>
        """)

        with gr.Row():
            # ── LEFT COL: Input & Examples ──
            with gr.Column(scale=4):
                input_box = gr.Textbox(
                    label="Ask a question",
                    placeholder="e.g. What is the PMS value of silver standard?",
                    lines=3,
                    autofocus=True,
                    elem_id="input-box"
                )
                
                with gr.Row():
                    submit_btn = gr.Button("🔍 Search SOP", variant="primary", scale=2)
                    clear_btn = gr.Button("🗑️ Clear", variant="secondary", scale=1)
                
                with gr.Accordion("⚙️ Options", open=False):
                    rewrite_toggle = gr.Checkbox(
                        label="Enable LLM Rewrite (requires Ollama)",
                        value=False
                    )

                # ── Examples Sections ──
                gr.Markdown("### 📋 Try these examples")
                with gr.Tabs():
                    with gr.Tab("🎨 Color & Ink"):
                        gr.Examples(examples_color, inputs=input_box, label=None)
                    with gr.Tab("📐 Artwork & files"):
                        gr.Examples(examples_artwork, inputs=input_box, label=None)
                    with gr.Tab("🔤 Typography"):
                        gr.Examples(examples_type, inputs=input_box, label=None)
                    with gr.Tab("🧪 Guardrail Tests"):
                        gr.Examples(examples_oos, inputs=input_box, label=None)

            # ── RIGHT COL: Results ──
            with gr.Column(scale=5):
                # Answer Box
                answer_result = gr.Textbox(
                    label="Answer", 
                    lines=4, 
                    elem_classes=["answer-box"]
                )
                
                # Consolidated Status
                status_html = gr.HTML(label="Analysis Status")

                # Debug Info (Hidden by default)
                with gr.Accordion("🛠️ Debug Information", open=False):
                    debug_info = gr.Textbox(
                        label="Pipeline Details",
                        lines=10,
                        elem_classes=["debug-box"]
                    )
                
                # System Status
                with gr.Accordion("🖥️ System Health", open=False):
                     system_stat = gr.Textbox(
                        show_label=False,
                        lines=6,
                        value=lambda: get_system_status()
                    )
                     refresh_sys = gr.Button("Refresh Status", size="sm")
                     refresh_sys.click(get_system_status, outputs=system_stat)

        # ── Logic ──────────────────────────────
        def process_query(q, rw):
            # Wrapper to parse generic output into specific UI fields
            ans, stat_text, match, score, dbg = answer_question(q, rw)
            
            # Reconstruct result dict for HTML formatter
            # (We need to reverse-engineer matching logic or modify answer_question to return dict)
            # Easier: modify answer_question? No, let's parse or just update answer_question in next step.
            # Using existing return values to build a fake dict for formatter:
            
            is_rejected = "REJECTED" in stat_text
            try:
                sc = float(score) if score != "N/A" else 0.0
            except: sc = 0.0
            
            fake_result = {
                "answer": ans.replace("✅ ", "").replace("❌ ", ""),
                "rejected": is_rejected,
                "score": sc,
                "matched_question": match,
                "threshold": 0.0 # Not passed, but HTML doesn't strictly need it if we put score
            }
            html = format_status_html(fake_result)
            return ans, html, dbg

        submit_btn.click(
            process_query,
            inputs=[input_box, rewrite_toggle],
            outputs=[answer_result, status_html, debug_info]
        )
        input_box.submit(
            process_query,
            inputs=[input_box, rewrite_toggle],
            outputs=[answer_result, status_html, debug_info]
        )
        clear_btn.click(
            lambda: ("", "", ""),
            outputs=[answer_result, status_html, debug_info]
        )

    return app


# ── Main ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  SOP Chatbot V5 — Starting Web Interface")
    print("=" * 60)

    # Initialize pipeline
    initialize_pipeline()

    # Create and launch app
    app = create_app()
    app.launch(
        server_name=config.APP_HOST,
        server_port=config.APP_PORT,
        share=False,
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(
            primary_hue=gr.themes.colors.indigo,
            secondary_hue=gr.themes.colors.purple,
            neutral_hue=gr.themes.colors.slate,
            font=gr.themes.GoogleFont("Inter"),
        ),
    )
