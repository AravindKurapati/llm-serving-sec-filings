import streamlit as st
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── After running `modal deploy finsight.py`, paste your URL here ──
MODAL_URL = "https://your-workspace--finsight-api-query.modal.run"

st.set_page_config(page_title="FinSight", layout="wide")
st.title("FinSight — SEC Filing Q&A")
st.caption("LLaMA 3.1 8B vs Mistral 7B on real 10-K filings (AAPL, MSFT, GOOGL, AMZN, META)")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    model = st.selectbox(
        "Model (Single-model tab)",
        ["llama", "mistral"],
        format_func=lambda x: "LLaMA 3.1 8B" if x == "llama" else "Mistral 7B",
    )
    max_tokens = st.slider("Max tokens", 100, 600, 400, step=50)
    k = st.slider("Retrieved chunks (k)", 3, 10, 5)

    st.divider()
    st.markdown("**About**")
    st.markdown(
        "Answers grounded in real SEC 10-K filings. "
        "Each answer includes source citations and latency metrics."
    )

    st.divider()
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

    # ── RAGAS Benchmark Panel ─────────────────────────────────────────────────
    st.divider()
    st.markdown("### RAGAS Benchmark")
    st.caption("Evaluated on 5 qualitative questions about SEC filings")

    ragas_data = {
        "LLaMA 3.1 8B": {
            "faithfulness":       0.0,
            "answer_relevancy":   0.87,
            "context_precision":  0.0,
        },
        "Mistral 7B": {
            "faithfulness":       0.0,
            "answer_relevancy":   0.84,
            "context_precision":  0.0,
        },
    }

    for mdl_name, scores in ragas_data.items():
        st.markdown(f"**{mdl_name}**")
        cols = st.columns(3)
        cols[0].metric("Faithfulness",      f"{scores['faithfulness']:.2f}")
        cols[1].metric("Ans. Relevancy",    f"{scores['answer_relevancy']:.2f}")
        cols[2].metric("Ctx. Precision",    f"{scores['context_precision']:.2f}")

    st.info(
        "**Why 0.0 on faithfulness & context precision?**  \n"
        "The test set included quantitative questions (e.g. exact revenue figures). "
        "RAGAS LLM-based metrics cannot verify numerical claims against retrieved "
        "passages, so they score 0. Answer relevancy (embedding-based) is unaffected.",
        icon=None,
    )

# ── Session state init ────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "compare_history" not in st.session_state:
    st.session_state.compare_history = []   # list of {question, llama, mistral}

# ── Helper ────────────────────────────────────────────────────────────────────
def query_model(question: str, mdl: str) -> dict:
    """POST to Modal backend; returns {answer, metrics, contexts} or error dict."""
    try:
        r = requests.post(
            MODAL_URL,
            json={"question": question, "model": mdl, "k": k, "max_tokens": max_tokens},
            timeout=180,
        )
        data = r.json()
        return {
            "answer":   data["answer"],
            "metrics":  data.get("metrics", {}),
            "contexts": data.get("contexts", []),
            "error":    None,
        }
    except Exception as e:
        return {"answer": f"Error: {e}", "metrics": {}, "contexts": [], "error": str(e)}


def render_metrics(metrics: dict, model_label: str) -> None:
    if not metrics:
        return
    st.caption(
        f"Model: {model_label} | "
        f"TTFT: {metrics['ttft_ms']} ms | "
        f"TPOT: {metrics['tpot_ms']} ms | "
        f"Throughput: {metrics['throughput_tps']} tok/s | "
        f"{metrics['tokens']} tokens"
    )


def render_sources(contexts: list) -> None:
    if not contexts:
        return
    with st.expander("View retrieved sources"):
        for i, ctx in enumerate(contexts):
            st.markdown(f"**[{i+1}] {ctx['src']}**")
            st.text(ctx["text"][:300] + "...")


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_single, tab_compare = st.tabs(["Single Model", "Side-by-Side Comparison"])

# ─── Tab 1: Single model chat ─────────────────────────────────────────────────
with tab_single:
    MODEL_LABEL = "LLaMA 3.1 8B" if model == "llama" else "Mistral 7B"

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if "metrics" in msg and msg["metrics"]:
                render_metrics(msg["metrics"], msg.get("model_label", MODEL_LABEL))
            if "contexts" in msg and msg["contexts"]:
                render_sources(msg["contexts"])

    if question := st.chat_input(
        "Ask about Apple, Microsoft, Google, Amazon, or Meta… "
        "(e.g. 'What are Apple's main supply chain risks?')"
    ):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner(f"Asking {MODEL_LABEL}…"):
                result = query_model(question, model)

            st.write(result["answer"])
            render_metrics(result["metrics"], MODEL_LABEL)
            render_sources(result["contexts"])

        st.session_state.messages.append({
            "role":        "assistant",
            "content":     result["answer"],
            "model_label": MODEL_LABEL,
            "metrics":     result["metrics"],
            "contexts":    result["contexts"],
        })

# ─── Tab 2: Side-by-side comparison ──────────────────────────────────────────
with tab_compare:
    st.markdown(
        "Ask one question and get both models' answers **simultaneously**, "
        "with latency metrics for direct comparison."
    )

    # Replay prior comparisons
    for entry in st.session_state.compare_history:
        st.markdown(f"**You:** {entry['question']}")
        col_l, col_r = st.columns(2)

        with col_l:
            st.subheader("LLaMA 3.1 8B")
            st.write(entry["llama"]["answer"])
            render_metrics(entry["llama"]["metrics"], "LLaMA 3.1 8B")
            render_sources(entry["llama"]["contexts"])

        with col_r:
            st.subheader("Mistral 7B")
            st.write(entry["mistral"]["answer"])
            render_metrics(entry["mistral"]["metrics"], "Mistral 7B")
            render_sources(entry["mistral"]["contexts"])

        st.divider()

    if cmp_question := st.chat_input(
        "Compare both models… (e.g. 'What risks does Meta face in its metaverse bet?')",
        key="compare_input",
    ):
        st.markdown(f"**You:** {cmp_question}")

        col_l, col_r = st.columns(2)
        spin_l = col_l.empty()
        spin_r = col_r.empty()

        with spin_l:
            st.spinner("Asking LLaMA 3.1 8B…")
        with spin_r:
            st.spinner("Asking Mistral 7B…")

        # Fire both requests in parallel
        results = {}
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {
                pool.submit(query_model, cmp_question, "llama"):   "llama",
                pool.submit(query_model, cmp_question, "mistral"): "mistral",
            }
            for fut in as_completed(futures):
                results[futures[fut]] = fut.result()

        spin_l.empty()
        spin_r.empty()

        with col_l:
            st.subheader("LLaMA 3.1 8B")
            st.write(results["llama"]["answer"])
            render_metrics(results["llama"]["metrics"], "LLaMA 3.1 8B")
            render_sources(results["llama"]["contexts"])

        with col_r:
            st.subheader("Mistral 7B")
            st.write(results["mistral"]["answer"])
            render_metrics(results["mistral"]["metrics"], "Mistral 7B")
            render_sources(results["mistral"]["contexts"])

        st.divider()

        st.session_state.compare_history.append({
            "question": cmp_question,
            "llama":    results["llama"],
            "mistral":  results["mistral"],
        })
