# global_utils.py
import streamlit as st
import os

def apply_global_styles():
    base_path = os.path.dirname(os.path.abspath(__file__))
    logo_path = os.path.join(base_path, "assets", "logo.png")
    
    if os.path.exists(logo_path):
        st.sidebar.image(logo_path, width='stretch')
    
    css_path = os.path.join(base_path, "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path, encoding='utf-8') as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.markdown("""
            <style>
            .main { padding: 2rem; }
            h1 { color: #008542; } /* Verde Petrobras */
            .stChatMessage { border-radius: 15px; }
            </style>
        """, unsafe_allow_html=True)