import streamlit as st
import requests

st.set_page_config(page_title="FinSight", page_icon="$")
st.title("FinSight - SEC Filing Q&A")

MODAL_URL = "https://arvind-kurapati--finsight-api-query.modal.run"

model = st.sidebar.selectbox("Model", ["llama", "mistral"])
st.sidebar.caption("LLaMA 3.1 8B vs Mistral 7B")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "metrics" in msg:
            m = msg["metrics"]
            st.caption(f"TTFT: {m['ttft_ms']}ms | TPOT: {m['tpot_ms']}ms | {m['throughput_tps']} tok/s")

if question := st.chat_input("Ask about Apple, Microsoft, Google, Amazon or Meta..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner(f"Asking {model}..."):
            try:
                r = requests.post(
                    MODAL_URL,
                    json={"question": question, "model": model},
                    timeout=120,
                )
                data = r.json()
                answer  = data["answer"]
                metrics = data["metrics"]
            except Exception as e:
                answer  = f"Error: {e}"
                metrics = {}

        st.write(answer)
        if metrics:
            st.caption(
                f"TTFT: {metrics['ttft_ms']}ms | "
                f"TPOT: {metrics['tpot_ms']}ms | "
                f"{metrics['throughput_tps']} tok/s | "
                f"Model: {model}"
            )
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "metrics": metrics,
        })
