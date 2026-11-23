import streamlit as st
from store import get_index
from llm import answer_question

st.set_page_config(page_title="Policy Chatbot", page_icon="💬")
st.title("💬 Company Policy Chatbot")
st.caption("Ask a question and I'll answer from the company policy document.")


@st.cache_resource
def load_index():
    """Connect to Pinecone once, then reuse the connection."""
    return get_index()


index = load_index()

# st.session_state remembers things between reruns of the script
if "messages" not in st.session_state:
    st.session_state.messages = []


def show_sources(sources):
    """Display the policy sections the answer was based on."""
    if sources:
        with st.expander("Sources"):
            for s in sources:
                page = f", page {int(s['page'])}" if s["page"] is not None else ""
                st.markdown(f"**{s['heading'] or s['source']}** ({s['source']}{page}) · score {s['score']:.2f}")
                st.text(s["text"])


# Redraw the whole conversation so far
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant":
            show_sources(message["sources"])

# The chat box at the bottom of the page
question = st.chat_input("Ask a question about company policy...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the policy..."):
            answer, sources = answer_question(question, index)
        st.write(answer)
        show_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )