import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from src.agent import Session, SupportAgent

"""ShopEase Customer Support Assistant - Streamlit UI."""
import streamlit as st

from src.agent import Session, SupportAgent
from src.config import load_settings
from src.embeddings import get_embedder
from src.ingest import ensure_index
from src.llm import LLM
from src.retrieval import KnowledgeBase
from src.tools import SupportTools

st.set_page_config(page_title="ShopEase Support", page_icon="🛍️")

SAMPLES = [
    "What is the return window for electronics?",
    "How much does expedited shipping cost?",
    "Where is my order SE-1002?",
    "I want to return the earbuds from SE-1001",
    "Can I return the tablet from SE-3001?",
    "What's the weather in Paris?",
]


@st.cache_resource(show_spinner="Loading knowledge base (the first run downloads the embedding model)…")
def load_stack():
    cfg = load_settings()
    emb = get_embedder(cfg)
    ensure_index(cfg, emb)
    kb = KnowledgeBase(cfg, emb)
    tools = SupportTools(cfg)
    return cfg, kb, tools, SupportAgent(cfg, kb, LLM(cfg), tools)


try:
    cfg, kb, tools, agent = load_stack()
except Exception as e:  # missing API key, invalid index, ...
    st.error(f"Startup problem: {e}")
    st.stop()

st.session_state.setdefault("messages", [])
st.session_state.setdefault("pending", None)
people = {"Guest (not signed in)": None} | {f"{c['name']} · {c['email']}": c["customer_id"] for c in tools.customers()}


def reset():
    st.session_state.messages, st.session_state.pending = [], None


def esc(text):  # Streamlit renders $...$ as LaTeX, so escape prices
    return text.replace("$", "\\$")


def show(m):
    with st.chat_message(m["role"]):
        st.markdown(esc(m["content"]))
        cites = m.get("citations") or []
        if cites:
            with st.expander(f"Sources ({len(cites)})"):
                for c in cites:
                    if c["type"] == "kb":
                        page = f" · p.{c['page']}" if c.get("page") else ""
                        st.markdown(f"**{c['label']}** {c['title']} › {c['section']}{page} · v{c['version']}  \n"
                                    f"`{c['filename']}` · `{c['chunk_id']}` · similarity {c['score']:.2f}")
                    else:
                        st.markdown(f"**{c['label']}** system tool `{c['tool']}` · `{c['args']}` · {c['as_of']}")
        if m.get("debug") and st.session_state.get("show_debug"):
            st.json(m["debug"])


with st.sidebar:
    st.subheader("Demo controls")
    st.selectbox("Sign in as (mock accounts)", list(people), key="who", on_change=reset)  # switching account clears the chat
    st.toggle("Show verification details", key="show_debug")
    st.button("Clear chat", on_click=reset)
    st.caption("Try one of these:")
    for s in SAMPLES:
        st.button(s, key=f"sample-{s}", on_click=lambda s=s: st.session_state.update(queued=s))
    st.caption(f"KB: {kb.info['documents']} docs · {kb.info['chunks']} chunks · {kb.info['embedding_model']}")

session = Session(customer_id=people[st.session_state.who])
st.title(cfg.app.name)
st.caption("Answers come only from the help-center knowledge base and your own account data, with sources. Actions need your confirmation.")

for m in st.session_state.messages:
    show(m)


def do_confirm():
    res = agent.confirm_action(st.session_state.pending, session)
    st.session_state.messages.append({"role": "assistant", "content": res.answer, "citations": res.citations})
    st.session_state.pending = None


def do_cancel():
    st.session_state.messages.append({"role": "assistant", "content": "Okay - I did not make any changes."})
    st.session_state.pending = None


def do_handoff():
    last_q = next((m["content"] for m in reversed(st.session_state.messages) if m["role"] == "user"), "")
    st.session_state.pending = {"tool": "escalate_to_human", "args": {"summary": last_q},
                                "summary": "Open a support ticket for a human agent"}


pending = st.session_state.pending
if pending:
    with st.container(border=True):
        st.markdown(f"**Please confirm:** {esc(pending['summary'])}")
        c1, c2 = st.columns(2)
        c1.button("✅ Confirm", type="primary", on_click=do_confirm)
        c2.button("Cancel", on_click=do_cancel)
elif st.session_state.messages and st.session_state.messages[-1].get("offer_handoff"):
    st.button("👤 Talk to a human agent", on_click=do_handoff)

prompt = st.chat_input("Ask about orders, returns, shipping, payments, Plus…") or st.session_state.pop("queued", None)
if prompt:
    history = list(st.session_state.messages)
    st.session_state.pending = None
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(esc(prompt))
    with st.chat_message("assistant"), st.spinner("Checking the knowledge base…"):
        res = agent.handle(prompt, history, session)
    st.session_state.messages.append({"role": "assistant", "content": res.answer, "citations": res.citations,
                                      "debug": res.debug, "offer_handoff": res.offer_handoff})
    st.session_state.pending = res.pending_action
    st.rerun()
