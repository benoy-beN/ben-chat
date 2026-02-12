"""
SOP Chatbot — Gradio Web Interface.
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
        status = f"🔴 REJECTED — Below threshold ({result['score']:.4f} < {result['threshold']})"
    else:
        answer = f"✅ {result['answer']}"
        status = f"🟢 MATCHED — Confidence: {result['score']:.4f} (threshold: {result['threshold']})"

    matched = result["matched_question"] or "No match found"
    score = f"{result['score']:.4f}" if result["score"] > 0 else "N/A"

    # Debug: top-K results
    debug_lines = [f"⏱️ Response time: {elapsed*1000:.1f}ms", ""]
    debug_lines.append("Top matches:")
    for i, r in enumerate(result.get("top_k_results", []), 1):
        marker = "→" if i == 1 else " "
        debug_lines.append(
            f"  {marker} [{i}] Score: {r['score']:.4f} | Q: {r['question']}"
        )
    debug_info = "\n".join(debug_lines)

    return answer, status, matched, score, debug_info


def get_system_status() -> str:
    """Get system status information."""
    status_lines = [
        f"📦 Embedding Model: {config.EMBEDDING_MODEL}",
        f"📊 Threshold: {config.SIMILARITY_THRESHOLD}",
        f"📂 SOP Entries (A): {pipeline.index_a.index.ntotal if pipeline and pipeline.index_a.index else 'N/A'}",
        f"🔍 SOP Entries (B): {pipeline.index_b.index.ntotal if pipeline and pipeline.index_b.index else 'N/A'}",
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
def create_app():
    """Create and configure the Gradio app."""

    with gr.Blocks(
        title="SOP Chatbot — Zero Hallucination",
    ) as app:

        # ── Header ──────────────────────────────
        gr.HTML("""
        <div class="main-header">
            <h1>🛡️ SOP Chatbot</h1>
            <p>Deterministic retrieval • Zero hallucination • Auditable answers</p>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=3):
                # ── Input ────────────────────────
                question_input = gr.Textbox(
                    label="Ask a question",
                    placeholder="e.g. What is the PMS value of silver standard?",
                    lines=2,
                    max_lines=4,
                    autofocus=True,
                )

                with gr.Row():
                    submit_btn = gr.Button(
                        "🔍 Search SOP",
                        variant="primary",
                        size="lg",
                    )
                    clear_btn = gr.Button(
                        "🗑️ Clear",
                        variant="secondary",
                        size="lg",
                    )
                    rewrite_toggle = gr.Checkbox(
                        label="✨ LLM Rewrite",
                        value=False,
                        info="Reformat answer using Ollama (requires Ollama running)",
                    )

                # ── Answer ───────────────────────
                answer_output = gr.Textbox(
                    label="Answer",
                    lines=3,
                    max_lines=8,
                    interactive=False,
                    elem_classes=["answer-box"],
                )

                # ── Status ───────────────────────
                status_output = gr.Textbox(
                    label="Status",
                    lines=1,
                    interactive=False,
                    elem_classes=["status-box"],
                )

            with gr.Column(scale=2):
                # ── Match Details ────────────────
                matched_output = gr.Textbox(
                    label="Matched SOP Question",
                    lines=2,
                    interactive=False,
                )
                score_output = gr.Textbox(
                    label="Similarity Score",
                    lines=1,
                    interactive=False,
                )

                # ── Debug ────────────────────────
                debug_output = gr.Textbox(
                    label="Debug Info",
                    lines=8,
                    interactive=False,
                    elem_classes=["debug-box"],
                )

                # ── System Status ────────────────
                with gr.Accordion("⚙️ System Status", open=False):
                    system_status = gr.Textbox(
                        label="",
                        lines=5,
                        interactive=False,
                        value=lambda: get_system_status(),
                    )
                    refresh_btn = gr.Button("🔄 Refresh", size="sm")
                    refresh_btn.click(fn=get_system_status, outputs=system_status)

        # ── Example Questions ────────────────────
        gr.Examples(
            examples=[
                ["What is the PMS value of silver standard?"],
                ["Should text be converted to outlines?"],
                ["What file formats are preferred for final vector artwork?"],
                ["How do I cook pasta?"],  # Out-of-scope — should reject
                ["What is the minimum font size for positive sans-serif text?"],
                ["Should artwork exceed the imprint area?"],
                ["Who is responsible for final artwork accuracy?"],
                ["What is the weather today?"],  # Out-of-scope — should reject
            ],
            inputs=question_input,
            label="📋 Try these examples",
        )

        # ── Event Handlers ───────────────────────
        submit_btn.click(
            fn=answer_question,
            inputs=[question_input, rewrite_toggle],
            outputs=[answer_output, status_output, matched_output, score_output, debug_output],
        )

        question_input.submit(
            fn=answer_question,
            inputs=[question_input, rewrite_toggle],
            outputs=[answer_output, status_output, matched_output, score_output, debug_output],
        )

        clear_btn.click(
            fn=lambda: ("", "", "", "", ""),
            outputs=[answer_output, status_output, matched_output, score_output, debug_output],
        )

    return app


# ── Main ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  SOP Chatbot — Starting Web Interface")
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
