#
# ./Home.py
#

import streamlit as st
import os
import datetime
from global_utils import apply_global_styles # Importe a função

# --- Configurações da Página ---
st.set_page_config(
    page_title="Hub de Dados e IA da ES",
    page_icon="🏠",
    layout="wide", # Usa a largura total da tela
    initial_sidebar_state="expanded" # Sidebar visível por padrão
)

st.sidebar.info("Selecione uma aplicação no menu acima para começar.")

# --- APLICA ESTILOS GLOBAIS E LOGO ---
apply_global_styles() # <-- Chame a função aqui, logo após st.set_page_config

# --- Cabeçalho ---
st.title("Bem-vindo(a) ao Hub de Soluções em Dados e IA da ES 🎲")
st.markdown("""
    Explore um mundo de possibilidades com nossas ferramentas de Inteligência Artificial! \n
    Soluções desenvolvidas com parceiros de toda a UO-ES para otimizar seus processos e gerar mais valor ao seu trabalho.
""")

# Pega o diretório atual do script (Início.py)
base_path = os.path.dirname(__file__)
logo_path = os.path.join(base_path, "assets", "logo.png")

# --- Sidebar para Navegação --- (tirei tudo, depois volto nele)
# st.sidebar.title("Navegação")
# st.sidebar.markdown("---")
# st.sidebar.info("Selecione uma aplicação no menu acima para começar.")

# --- Conteúdo Principal (placeholder para futuras informações) ---
st.write("---")
st.subheader("O que esperar?")
st.write("""
    Nossas soluções de IA são projetadas para serem intuitivas e eficazes.
    Você encontrará ferramentas para processamento de documentos, análise de dados,
    geração de conteúdo e muito mais.
""")

# Obtenha o ano atual
current_year = datetime.datetime.now().year

# Use o ano na string do caption
st.markdown("---")
st.caption(f"Coordenação de Soluções Digitais e Analíticas da ES/ENGP/CSDA - {current_year}")