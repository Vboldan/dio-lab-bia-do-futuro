import os
import json
import pandas as pd
import streamlit as st
import openai
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    st.warning("Defina OPENAI_API_KEY no .env ou nas variáveis de ambiente.")
else:
    openai.api_key = OPENAI_KEY

@st.cache_data
def load_transactions(path="data/transacoes.csv"):
    try:
        df = pd.read_csv(path, parse_dates=["data"])    
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data
def load_profile(path="data/perfil_investidor.json"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

transactions = load_transactions()
profile = load_profile()

st.set_page_config(page_title="Agente Financeiro - Protótipo", layout="wide")
st.title("🤖 Agente Financeiro — Protótipo")

with st.sidebar:
    st.header("Config")
    st.write("Modelo LLM usado: gpt-3.5-turbo (pode ser alterado)")
    temperature = st.slider("Temperatura (exploração)", min_value=0.0, max_value=1.0, value=0.2)

with st.expander("Perfil do cliente"):
    st.json(profile)

with st.expander("Últimas transações"):
    if transactions.empty:
        st.info("Nenhuma transação encontrada em data/transacoes.csv")
    else:
        st.dataframe(transactions.sort_values("data", ascending=False).head(50))

# Inicializa histórico de mensagens
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "Você é um assistente financeiro em português. Responda de forma curta, precisa e cite quando não houver informação nos dados. Se houver incerteza, diga 'Não sei' ou solicite mais dados."}
    ]

# Função para consulta ao LLM
def query_llm(user_message: str):
    st.session_state.messages.append({"role": "user", "content": user_message})
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=st.session_state.messages,
            temperature=float(temperature),
            max_tokens=512,
        )
        content = resp.choices[0].message["content"]
    except Exception as e:
        content = f"Erro ao chamar LLM: {e}"
    st.session_state.messages.append({"role": "assistant", "content": content})
    return content

# UI do chat
st.markdown("---")
st.subheader("Chat com o agente")
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Pergunte algo (ex: \"Resumo dos gastos do mês\", \"Sugestão de meta de economia\"):")
    submitted = st.form_submit_button("Enviar")
    if submitted and user_input:
        with st.spinner("Consultando agente..."):
            answer = query_llm(user_input)
            st.markdown("**Resposta do agente:**")
            st.write(answer)

# Exibe histórico local
st.markdown("---")
st.markdown("### Histórico (sessão)")
for m in st.session_state.messages:
    if m["role"] == "user":
        st.markdown(f"**Você:** {m['content']}")
    elif m["role"] == "assistant":
        st.markdown(f"**Agente:** {m['content']}")

# Footer com recomendações de segurança
st.markdown("---")
st.caption("Dica: Em produção, use RAG (recuperação+fusão) para alimentar o modelo apenas com trechos relevantes, e mantenha temperatura baixa (0.0-0.3) para minimizar alucinações.")
