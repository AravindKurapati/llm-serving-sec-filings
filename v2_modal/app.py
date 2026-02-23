import streamlit as st
import requests

# ── After running `modal deploy finsight.py`, paste your URL here ──
MODAL_URL = "https://your-workspace--finsight-api-query.modal.run"

st.set_page_config(page_title="FinSight", page_icon="📊", layout="wide")
st.title("📊 FinSight — SEC Filing Q&A")
st.caption("LLaMA 3.1 8B vs Mistral 7B on real 10-K filings (AAPL, MSFT, GOOGL, AMZN, META)")

# Sidebar
with st.sidebar:
    st.header("Settings")
    model = st.selectbox("Model", ["llama", "mistral"],
                         format_func=lambda x: "LLaMA 3.1 8B" if x == "llama" else "Mistral 7B")
    max_tokens = st.slider("Max tokens", 100, 600, 400, step=50)
    k = st.slider("Retrieved chunks (k)", 3, 10, 5)
    st.divider()
    st.markdown("**About**")
    st.markdown("Answers grounded in real SEC 10-K filings. "
                "Each answer includes source citations and latency metrics.")
    st.divider()
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "metrics" in msg and msg["metrics"]:
            m = msg["metrics"]
            st.caption(
                f"Model: {msg.get('model', model)} | "
                f"TTFT: {m['ttft_ms']}ms | "
                f"TPOT: {m['tpot_ms']}ms | "
                f"Throughput: {m['throughput_tps']} tok/s | "
                f"{m['tokens']} tokens"
            )
        if "contexts" in msg and msg["contexts"]:
            with st.expander("View retrieved sources"):
                for i, ctx in enumerate(msg["contexts"]):
                    st.markdown(f"**[{i+1}] {ctx['src']}**")
                    st.text(ctx["text"][:300] + "...")

# Chat input
if question := st.chat_input(
    "Ask about Apple, Microsoft, Google, Amazon, or Meta... "
    "(e.g. 'What are Apple's main supply chain risks?')"
):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner(f"Asking {model}..."):
            try:
                r = requests.post(
                    MODAL_URL,
                    json={
                        "question": question,
                        "model": model,
                        "k": k,
                        "max_tokens": max_tokens,
                    },
                    timeout=180,
                )
                data    = r.json()
                answer  = data["answer"]
                metrics = data.get("metrics", {})
                contexts = data.get("contexts", [])
            except Exception as e:
                answer   = f"Error: {e}"
                metrics  = {}
                contexts = []

        st.write(answer)

        if metrics:
            st.caption(
                f"Model: {model} | "
                f"TTFT: {metrics['ttft_ms']}ms | "
                f"TPOT: {metrics['tpot_ms']}ms | "
                f"Throughput: {metrics['throughput_tps']} tok/s | "
                f"{metrics['tokens']} tokens"
            )

        if contexts:
            with st.expander("View retrieved sources"):
                for i, ctx in enumerate(contexts):
                    st.markdown(f"**[{i+1}] {ctx['src']}**")
                    st.text(ctx["text"][:300] + "...")

    st.session_state.messages.append({
        "role":     "assistant",
        "content":  answer,
        "model":    model,
        "metrics":  metrics,
        "contexts": contexts,
    })
