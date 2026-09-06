import sys
import os
import time
import math
import json
from typing import Callable, Dict, Any

import streamlit as st
import pandas as pd
import folium
import osmnx as ox
import networkx as nx
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from streamlit_folium import st_folium


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# IMPORT PROJECT MODULES
# ============================================================

from graph.osm_loader import load_city_graph
from graph.graph_builder import describe_graph, validate_graph
from graph.location_snapper import snap_locations

from algorithms.qpso import QPSO_VRP
from algorithms.pso import PSO_VRP
from algorithms.ga import GeneticAlgorithmVRP
from algorithms.aco import ACOVRP
from algorithms.simulated_annealing import SimulatedAnnealingVRP
from algorithms.tabu_search import TabuSearchVRP
from algorithms.alns import ALNS_VRP
from traffic.dynamic_weights import DynamicWeights


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="FluxRoute — Intelligent Traffic Routing | SIH26137",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PREMIUM DARK GLASSMORPHISM THEME — HACKATHON EDITION
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ---- ROOT VARIABLES ---- */
:root {
    --bg-primary: #050510;
    --bg-secondary: #0a0a1e;
    --bg-tertiary: #0f0f2d;
    --bg-card: rgba(15, 15, 45, 0.6);
    --bg-glass: rgba(255, 255, 255, 0.02);
    --bg-glass-hover: rgba(255, 255, 255, 0.04);
    --border-glass: rgba(255, 255, 255, 0.06);
    --border-glow: rgba(0, 212, 255, 0.3);
    --border-glow-purple: rgba(124, 58, 237, 0.3);
    --accent-primary: #00d4ff;
    --accent-secondary: #7c3aed;
    --accent-tertiary: #a855f7;
    --accent-success: #00ff88;
    --accent-warning: #ffaa00;
    --accent-danger: #ff4466;
    --accent-info: #38bdf8;
    --text-primary: #e8ecf4;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --text-bright: #f8fafc;
    --shadow-glow: 0 0 40px rgba(0, 212, 255, 0.06);
    --shadow-card: 0 4px 30px rgba(0, 0, 0, 0.3);
    --radius-lg: 20px;
    --radius-md: 14px;
    --radius-sm: 10px;
}

/* ---- GLOBAL OVERRIDES ---- */
html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

[data-testid="stAppViewContainer"] > .main {
    background: transparent !important;
}

.block-container {
    padding-top: 1rem;
    padding-left: 2rem;
    padding-right: 2rem;
    padding-bottom: 2rem;
    max-width: 100%;
}

/* ---- ANIMATED BACKGROUND GRADIENT ---- */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background:
        radial-gradient(ellipse 800px 600px at 20% 20%, rgba(0, 212, 255, 0.04) 0%, transparent 70%),
        radial-gradient(ellipse 600px 500px at 80% 80%, rgba(124, 58, 237, 0.04) 0%, transparent 70%),
        radial-gradient(ellipse 500px 400px at 50% 50%, rgba(0, 255, 136, 0.02) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
    animation: bgPulse 8s ease-in-out infinite alternate;
}

@keyframes bgPulse {
    0% { opacity: 0.6; }
    50% { opacity: 1; }
    100% { opacity: 0.8; }
}

/* ---- SCROLLBAR ---- */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: rgba(0, 212, 255, 0.2);
    border-radius: 10px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(0, 212, 255, 0.4);
}

/* ---- SIDEBAR ---- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060618 0%, #0a0a24 40%, #0d0d2a 100%) !important;
    border-right: 1px solid var(--border-glass) !important;
    box-shadow: 4px 0 30px rgba(0, 0, 0, 0.4);
}

section[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}

section[data-testid="stSidebar"] .stTextInput > div > div > input {
    background: rgba(255, 255, 255, 0.04) !important;
    border: 1px solid var(--border-glass) !important;
    color: var(--text-primary) !important;
    border-radius: var(--radius-sm) !important;
    transition: all 0.3s ease;
}

section[data-testid="stSidebar"] .stTextInput > div > div > input:focus {
    border-color: var(--accent-primary) !important;
    box-shadow: 0 0 20px rgba(0, 212, 255, 0.12), inset 0 0 20px rgba(0, 212, 255, 0.04) !important;
}

/* ---- SIDEBAR SECTION DIVIDERS ---- */
.sidebar-section {
    background: linear-gradient(135deg, rgba(0, 212, 255, 0.04), rgba(124, 58, 237, 0.04));
    border: 1px solid var(--border-glass);
    border-radius: var(--radius-md);
    padding: 14px 16px;
    margin: 8px 0;
}

.sidebar-section-title {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.8px;
    color: var(--text-muted);
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.sidebar-stat {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.03);
}

.sidebar-stat:last-child { border-bottom: none; }

.sidebar-stat-label {
    font-size: 0.78rem;
    color: var(--text-muted);
}

.sidebar-stat-value {
    font-size: 0.85rem;
    font-weight: 700;
    color: var(--accent-primary);
    font-family: 'JetBrains Mono', monospace;
}

/* ---- HERO HEADER ---- */
.hero-container {
    background: linear-gradient(135deg, rgba(0, 212, 255, 0.06) 0%, rgba(124, 58, 237, 0.06) 50%, rgba(0, 255, 136, 0.03) 100%);
    border: 1px solid var(--border-glass);
    border-radius: var(--radius-lg);
    padding: 32px 40px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
}

.hero-container::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg,
        transparent,
        var(--accent-primary),
        var(--accent-secondary),
        var(--accent-primary),
        transparent
    );
    background-size: 300% 100%;
    animation: shimmer 4s linear infinite;
}

.hero-container::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg,
        transparent,
        rgba(0, 212, 255, 0.2),
        transparent
    );
}

@keyframes shimmer {
    0% { background-position: 300% 0; }
    100% { background-position: -300% 0; }
}

.hero-eyebrow {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 3px;
    color: var(--accent-primary);
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 8px;
    opacity: 0.9;
}

.hero-eyebrow-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--accent-primary);
    animation: dotPulse 2s ease-in-out infinite;
}

@keyframes dotPulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 6px var(--accent-primary); }
    50% { opacity: 0.4; box-shadow: 0 0 12px var(--accent-primary); }
}

.hero-title {
    font-size: 2.8rem;
    font-weight: 900;
    letter-spacing: -2px;
    background: linear-gradient(135deg, #00d4ff 0%, #7c3aed 40%, #a855f7 70%, #00d4ff 100%);
    background-size: 300% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: gradientFlow 6s ease infinite;
    margin: 0;
    line-height: 1.1;
}

.brand-lockup {
    display: inline-flex;
    align-items: center;
    gap: 14px;
}

.brand-mark {
    width: 48px;
    height: 48px;
    position: relative;
    display: grid;
    place-items: center;
    border: 1px solid rgba(0, 212, 255, 0.65);
    border-radius: 15px 15px 15px 4px;
    background: linear-gradient(145deg, rgba(0, 212, 255, 0.22), rgba(124, 58, 237, 0.28));
    box-shadow: 0 0 24px rgba(0, 212, 255, 0.18), inset 0 0 18px rgba(255, 255, 255, 0.08);
    transform: rotate(-8deg);
}

.brand-mark::after {
    content: '';
    position: absolute;
    width: 9px;
    height: 9px;
    right: 5px;
    top: 5px;
    border-radius: 50%;
    background: var(--accent-success);
    box-shadow: 0 0 12px var(--accent-success);
}

.brand-mark span {
    color: var(--text-bright);
    font-size: 1.7rem;
    font-weight: 900;
    line-height: 1;
    transform: rotate(8deg);
}

.brand-wordmark {
    color: var(--text-bright);
    font-size: 1.1rem;
    font-weight: 900;
    letter-spacing: 2px;
    line-height: 1;
}

.brand-kicker {
    color: var(--text-muted);
    font-size: 0.58rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    margin-top: 6px;
    text-transform: uppercase;
}

@keyframes gradientFlow {
    0%, 100% { background-position: 0% center; }
    50% { background-position: 100% center; }
}

.hero-subtitle {
    color: var(--text-secondary);
    font-size: 1rem;
    font-weight: 400;
    margin-top: 8px;
    letter-spacing: 0.3px;
    line-height: 1.5;
}

.hero-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 14px;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: linear-gradient(135deg, rgba(0, 212, 255, 0.1), rgba(124, 58, 237, 0.1));
    border: 1px solid rgba(0, 212, 255, 0.2);
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 0.7rem;
    font-weight: 600;
    color: var(--accent-primary);
    letter-spacing: 0.8px;
    transition: all 0.3s ease;
}

.hero-badge:hover {
    border-color: var(--accent-primary);
    box-shadow: 0 0 16px rgba(0, 212, 255, 0.15);
    transform: translateY(-1px);
}

.hero-badge-purple {
    background: linear-gradient(135deg, rgba(124, 58, 237, 0.1), rgba(168, 85, 247, 0.1));
    border-color: rgba(124, 58, 237, 0.25);
    color: var(--accent-tertiary);
}

.hero-badge-green {
    background: linear-gradient(135deg, rgba(0, 255, 136, 0.08), rgba(0, 212, 255, 0.08));
    border-color: rgba(0, 255, 136, 0.2);
    color: var(--accent-success);
}

/* ---- STATUS BADGE ---- */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(0, 255, 136, 0.06);
    border: 1px solid rgba(0, 255, 136, 0.15);
    border-radius: 24px;
    padding: 8px 18px;
    font-size: 0.75rem;
    font-weight: 700;
    color: var(--accent-success);
    letter-spacing: 1px;
    text-transform: uppercase;
}

.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--accent-success);
    animation: statusPulse 2s ease-in-out infinite;
    box-shadow: 0 0 8px var(--accent-success);
}

@keyframes statusPulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.85); }
}

/* ---- GLASS CARDS ---- */
.glass-card {
    background: linear-gradient(135deg, rgba(15, 15, 45, 0.5), rgba(10, 10, 30, 0.8));
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--border-glass);
    border-radius: var(--radius-lg);
    padding: 24px 28px;
    margin-bottom: 16px;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

.glass-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.06), transparent);
}

.glass-card:hover {
    border-color: var(--border-glow);
    box-shadow: var(--shadow-glow), var(--shadow-card);
    transform: translateY(-2px);
}

/* ---- METRIC CARD ---- */
.metric-card {
    background: linear-gradient(145deg, rgba(0, 212, 255, 0.03) 0%, rgba(124, 58, 237, 0.03) 100%);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--border-glass);
    border-radius: var(--radius-lg);
    padding: 24px 20px;
    text-align: center;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 50%; transform: translateX(-50%);
    width: 60%;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0, 212, 255, 0.3), transparent);
}

.metric-card:hover {
    border-color: var(--border-glow);
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(0, 212, 255, 0.08);
}

.metric-icon {
    font-size: 2rem;
    margin-bottom: 10px;
    filter: drop-shadow(0 0 8px rgba(0, 212, 255, 0.3));
}

.metric-value {
    font-size: 1.8rem;
    font-weight: 900;
    color: var(--accent-primary);
    margin: 6px 0;
    font-family: 'JetBrains Mono', monospace;
    text-shadow: 0 0 20px rgba(0, 212, 255, 0.2);
}

.metric-value-small {
    font-size: 1.2rem;
}

.metric-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1.2px;
}

.metric-delta {
    font-size: 0.7rem;
    color: var(--accent-success);
    margin-top: 4px;
    font-weight: 600;
}

/* ---- METRIC COLOR VARIANTS ---- */
.metric-card.purple .metric-value { color: var(--accent-secondary); text-shadow: 0 0 20px rgba(124, 58, 237, 0.2); }
.metric-card.purple::before { background: linear-gradient(90deg, transparent, rgba(124, 58, 237, 0.3), transparent); }
.metric-card.green .metric-value { color: var(--accent-success); text-shadow: 0 0 20px rgba(0, 255, 136, 0.2); }
.metric-card.green::before { background: linear-gradient(90deg, transparent, rgba(0, 255, 136, 0.3), transparent); }
.metric-card.amber .metric-value { color: var(--accent-warning); text-shadow: 0 0 20px rgba(255, 170, 0, 0.2); }
.metric-card.amber::before { background: linear-gradient(90deg, transparent, rgba(255, 170, 0, 0.3), transparent); }

/* ---- ROUTE CARD ---- */
.route-card {
    background: linear-gradient(135deg, rgba(15, 15, 45, 0.5), rgba(10, 10, 30, 0.7));
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--border-glass);
    border-radius: var(--radius-lg);
    padding: 22px 26px;
    margin-bottom: 14px;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    border-left: 4px solid var(--accent-primary);
    position: relative;
    overflow: hidden;
}

.route-card::after {
    content: '';
    position: absolute;
    top: 0; right: 0;
    width: 150px;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(0, 212, 255, 0.02));
    pointer-events: none;
}

.route-card:hover {
    border-color: var(--border-glow);
    box-shadow: 0 8px 32px rgba(0, 212, 255, 0.06);
    transform: translateX(4px);
}

.route-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 14px;
}

.route-title {
    font-size: 1.05rem;
    font-weight: 800;
    color: var(--text-bright);
    display: flex;
    align-items: center;
    gap: 10px;
}

.route-vehicle-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    border-radius: 10px;
    font-size: 0.85rem;
    font-weight: 800;
    color: white;
    font-family: 'JetBrains Mono', monospace;
}

.route-stats {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: 12px;
    margin-bottom: 14px;
}

.route-stat {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.04);
    border-radius: var(--radius-sm);
    padding: 10px 12px;
    text-align: center;
}

.route-stat-value {
    font-size: 0.95rem;
    font-weight: 800;
    color: var(--text-bright);
    font-family: 'JetBrains Mono', monospace;
}

.route-stat-label {
    font-size: 0.68rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-top: 2px;
}

.route-path {
    font-size: 0.78rem;
    color: var(--text-muted);
    background: rgba(0, 0, 0, 0.2);
    border: 1px solid rgba(255, 255, 255, 0.04);
    border-radius: var(--radius-sm);
    padding: 12px 16px;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.3px;
    line-height: 1.6;
    overflow-x: auto;
}

.route-path .arrow {
    color: var(--accent-primary);
    font-weight: 700;
}

.route-path .depot {
    color: var(--accent-danger);
    font-weight: 700;
}

.route-path .customer {
    color: var(--text-primary);
    font-weight: 600;
}

/* ---- UTILIZATION BAR ---- */
.util-bar-container {
    background: rgba(255, 255, 255, 0.04);
    border-radius: 8px;
    height: 8px;
    margin-top: 6px;
    overflow: hidden;
    position: relative;
}

.util-bar-fill {
    height: 100%;
    border-radius: 8px;
    transition: width 1s ease;
    position: relative;
    overflow: hidden;
}

.util-bar-fill::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.15) 50%, transparent 100%);
    animation: utilShimmer 2s ease infinite;
}

@keyframes utilShimmer {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
}

/* ---- SECTION HEADER ---- */
.section-header {
    display: flex;
    align-items: center;
    gap: 14px;
    margin: 32px 0 18px 0;
    position: relative;
}

.section-icon {
    width: 40px;
    height: 40px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
    background: linear-gradient(135deg, rgba(0, 212, 255, 0.1), rgba(124, 58, 237, 0.1));
    border: 1px solid rgba(0, 212, 255, 0.15);
    box-shadow: 0 0 16px rgba(0, 212, 255, 0.05);
}

.section-title {
    font-size: 1.4rem;
    font-weight: 800;
    color: var(--text-bright);
    letter-spacing: -0.5px;
}

.section-subtitle {
    font-size: 0.82rem;
    color: var(--text-muted);
    margin-top: -6px;
    margin-bottom: 14px;
    margin-left: 54px;
}

/* ---- BUTTONS ---- */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
    border-radius: var(--radius-md) !important;
    padding: 14px 28px !important;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
    font-family: 'Inter', sans-serif !important;
    letter-spacing: 0.3px !important;
    box-shadow: 0 4px 16px rgba(0, 212, 255, 0.15) !important;
}

.stButton > button[kind="primary"]:hover {
    box-shadow: 0 8px 32px rgba(0, 212, 255, 0.3) !important;
    transform: translateY(-2px) !important;
}

.stButton > button[kind="secondary"], .stButton > button:not([kind]) {
    background: rgba(255, 255, 255, 0.04) !important;
    border: 1px solid var(--border-glass) !important;
    color: var(--text-primary) !important;
    border-radius: var(--radius-md) !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.3s ease !important;
}

.stButton > button[kind="secondary"]:hover, .stButton > button:not([kind]):hover {
    border-color: var(--border-glow) !important;
    box-shadow: 0 4px 16px rgba(0, 212, 255, 0.08) !important;
}

/* ---- SELECT, INPUT ---- */
.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stTextInput > div > div > input {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid var(--border-glass) !important;
    color: var(--text-primary) !important;
    border-radius: var(--radius-sm) !important;
    transition: all 0.3s ease !important;
}

.stSelectbox > div > div:hover,
.stNumberInput > div > div > input:hover {
    border-color: rgba(0, 212, 255, 0.15) !important;
}

/* ---- SLIDER ---- */
.stSlider > div > div > div > div {
    background: var(--accent-primary) !important;
}

.stSlider > div > div > div {
    background: rgba(255, 255, 255, 0.06) !important;
}

/* ---- DATAFRAME ---- */
.stDataFrame {
    border-radius: var(--radius-md) !important;
    overflow: hidden;
    border: 1px solid var(--border-glass) !important;
}

/* ---- EXPANDER ---- */
.streamlit-expanderHeader {
    background: rgba(255, 255, 255, 0.02) !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-primary) !important;
    font-weight: 600 !important;
}

/* ---- TABS ---- */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    gap: 8px;
    border-bottom: 1px solid var(--border-glass);
}

.stTabs [data-baseweb="tab"] {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: var(--radius-sm) var(--radius-sm) 0 0 !important;
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    transition: all 0.3s ease !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(0, 212, 255, 0.1), rgba(124, 58, 237, 0.1)) !important;
    border-color: rgba(0, 212, 255, 0.25) !important;
    color: var(--accent-primary) !important;
    font-weight: 700 !important;
}

/* ---- PROGRESS BAR ---- */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, var(--accent-primary), var(--accent-secondary)) !important;
    border-radius: 8px !important;
}

/* ---- SUCCESS/ERROR/WARNING/INFO ALERTS ---- */
.stAlert {
    border-radius: var(--radius-md) !important;
    backdrop-filter: blur(8px) !important;
}

/* ---- DOWNLOAD BUTTON ---- */
.stDownloadButton > button {
    background: rgba(0, 212, 255, 0.06) !important;
    border: 1px solid rgba(0, 212, 255, 0.15) !important;
    color: var(--accent-primary) !important;
    border-radius: var(--radius-md) !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
}

.stDownloadButton > button:hover {
    background: rgba(0, 212, 255, 0.12) !important;
    box-shadow: 0 4px 20px rgba(0, 212, 255, 0.12) !important;
    transform: translateY(-1px) !important;
}

/* ---- DIVIDER ---- */
hr {
    border-color: var(--border-glass) !important;
    opacity: 0.5;
}

/* ---- METRIC WIDGET OVERRIDE ---- */
[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid var(--border-glass);
    border-radius: var(--radius-md);
    padding: 16px;
}

[data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
}

[data-testid="stMetricValue"] {
    color: var(--accent-primary) !important;
    font-weight: 800 !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ---- ALGO INFO CARDS ---- */
.algo-card {
    background: linear-gradient(145deg, rgba(0, 212, 255, 0.03), rgba(124, 58, 237, 0.03));
    border: 1px solid var(--border-glass);
    border-radius: var(--radius-lg);
    padding: 24px;
    height: 100%;
    transition: all 0.4s ease;
    position: relative;
    overflow: hidden;
}

.algo-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: var(--radius-lg) var(--radius-lg) 0 0;
}

.algo-card:hover {
    border-color: var(--border-glow);
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(0, 212, 255, 0.06);
}

.algo-card-qpso::before { background: linear-gradient(90deg, var(--accent-primary), var(--accent-secondary)); }
.algo-card-pso::before { background: linear-gradient(90deg, var(--accent-secondary), var(--accent-tertiary)); }
.algo-card-ga::before { background: linear-gradient(90deg, var(--accent-success), var(--accent-primary)); }

.algo-card-icon {
    font-size: 2.2rem;
    margin-bottom: 12px;
}

.algo-card-title {
    font-size: 1.05rem;
    font-weight: 800;
    margin-bottom: 10px;
    letter-spacing: -0.3px;
}

.algo-card-title-qpso { color: var(--accent-primary); }
.algo-card-title-pso { color: var(--accent-secondary); }
.algo-card-title-ga { color: var(--accent-success); }

.algo-card-desc {
    font-size: 0.82rem;
    color: var(--text-secondary);
    line-height: 1.6;
}

.algo-card-strength {
    margin-top: 12px;
    padding-top: 12px;
    border-top: 1px solid rgba(255, 255, 255, 0.04);
    font-size: 0.78rem;
}

.algo-card-strength b {
    color: var(--text-bright);
}

/* ---- LANDING PLACEHOLDER ---- */
.landing-container {
    text-align: center;
    padding: 80px 40px;
    position: relative;
}

.landing-icon {
    font-size: 5rem;
    margin-bottom: 20px;
    filter: drop-shadow(0 0 30px rgba(0, 212, 255, 0.3));
    animation: landingFloat 3s ease-in-out infinite;
}

@keyframes landingFloat {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-12px); }
}

.landing-title {
    font-size: 1.6rem;
    font-weight: 800;
    color: var(--text-bright);
    margin-bottom: 10px;
    letter-spacing: -0.5px;
}

.landing-desc {
    color: var(--text-secondary);
    font-size: 0.95rem;
    max-width: 500px;
    margin: 0 auto;
    line-height: 1.6;
}

.landing-steps {
    display: flex;
    justify-content: center;
    gap: 32px;
    margin-top: 32px;
    flex-wrap: wrap;
}

.landing-step {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid var(--border-glass);
    border-radius: var(--radius-md);
    padding: 20px 24px;
    width: 180px;
    text-align: center;
    transition: all 0.3s ease;
}

.landing-step:hover {
    border-color: var(--border-glow);
    transform: translateY(-3px);
}

.landing-step-num {
    font-size: 1.6rem;
    font-weight: 900;
    color: var(--accent-primary);
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 8px;
}

.landing-step-text {
    font-size: 0.8rem;
    color: var(--text-secondary);
}

/* ---- FOOTER ---- */
.footer-container {
    background: linear-gradient(135deg, rgba(0, 212, 255, 0.03) 0%, rgba(124, 58, 237, 0.03) 100%);
    border: 1px solid var(--border-glass);
    border-radius: var(--radius-lg);
    padding: 28px 36px;
    text-align: center;
    margin-top: 40px;
    position: relative;
    overflow: hidden;
}

.footer-container::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0, 212, 255, 0.2), rgba(124, 58, 237, 0.2), transparent);
}

.footer-brand {
    font-size: 1.1rem;
    font-weight: 900;
    background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.5px;
}

.footer-text {
    color: var(--text-muted);
    font-size: 0.75rem;
    margin-top: 6px;
    letter-spacing: 0.3px;
}

.footer-links {
    display: flex;
    justify-content: center;
    gap: 16px;
    margin-top: 12px;
}

.footer-link {
    font-size: 0.72rem;
    color: var(--text-muted);
    padding: 4px 12px;
    border: 1px solid rgba(255, 255, 255, 0.04);
    border-radius: 16px;
    transition: all 0.3s ease;
}

.footer-link:hover {
    border-color: var(--border-glow);
    color: var(--accent-primary);
}

/* ---- BENCHMARK TABLE HIGHLIGHT ---- */
.benchmark-winner {
    background: linear-gradient(135deg, rgba(0, 255, 136, 0.06), rgba(0, 212, 255, 0.06));
    border: 1px solid rgba(0, 255, 136, 0.15);
    border-radius: var(--radius-md);
    padding: 16px 20px;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 14px;
}

.benchmark-winner-icon {
    font-size: 2rem;
    filter: drop-shadow(0 0 12px rgba(255, 215, 0, 0.4));
}

.benchmark-winner-text {
    font-size: 0.9rem;
    color: var(--text-bright);
    font-weight: 700;
}

.benchmark-winner-detail {
    font-size: 0.78rem;
    color: var(--accent-success);
    margin-top: 2px;
    font-family: 'JetBrains Mono', monospace;
}

/* ---- FLOATING PARTICLES (CSS-only) ---- */
.particles {
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    pointer-events: none;
    z-index: 0;
    overflow: hidden;
}

.particle {
    position: absolute;
    width: 3px;
    height: 3px;
    border-radius: 50%;
    background: var(--accent-primary);
    opacity: 0;
    animation: particleFloat linear infinite;
}

.particle:nth-child(1) { left: 10%; animation-duration: 25s; animation-delay: 0s; }
.particle:nth-child(2) { left: 25%; animation-duration: 20s; animation-delay: 3s; }
.particle:nth-child(3) { left: 40%; animation-duration: 28s; animation-delay: 1s; }
.particle:nth-child(4) { left: 55%; animation-duration: 22s; animation-delay: 5s; }
.particle:nth-child(5) { left: 70%; animation-duration: 26s; animation-delay: 2s; }
.particle:nth-child(6) { left: 85%; animation-duration: 24s; animation-delay: 4s; }
.particle:nth-child(7) { left: 15%; animation-duration: 30s; animation-delay: 6s; background: var(--accent-secondary); }
.particle:nth-child(8) { left: 50%; animation-duration: 23s; animation-delay: 7s; background: var(--accent-secondary); }
.particle:nth-child(9) { left: 35%; animation-duration: 27s; animation-delay: 8s; }
.particle:nth-child(10) { left: 75%; animation-duration: 21s; animation-delay: 9s; background: var(--accent-success); }

@keyframes particleFloat {
    0% {
        transform: translateY(100vh) scale(0);
        opacity: 0;
    }
    10% { opacity: 0.6; }
    90% { opacity: 0.4; }
    100% {
        transform: translateY(-10vh) scale(1);
        opacity: 0;
    }
}

/* ---- HIDE STREAMLIT DEFAULTS ---- */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>

<!-- Floating Particles -->
<div class="particles">
    <div class="particle"></div>
    <div class="particle"></div>
    <div class="particle"></div>
    <div class="particle"></div>
    <div class="particle"></div>
    <div class="particle"></div>
    <div class="particle"></div>
    <div class="particle"></div>
    <div class="particle"></div>
    <div class="particle"></div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# BENCHMARK ENGINE
# ============================================================

class BenchmarkResult:
    """Stores the result of one optimization algorithm."""

    def __init__(self, algorithm_name, routes, distance, runtime,
                 iterations=0, history=None):
        self.algorithm_name = algorithm_name
        self.routes = routes
        self.distance = float(distance)
        self.runtime = float(runtime)
        self.iterations = int(iterations)
        self.history = history or []

    def to_dict(self):
        return {
            "Algorithm": self.algorithm_name,
            "Total Distance (km)": round(self.distance, 2),
            "Runtime (seconds)": round(self.runtime, 4),
            "Iterations": self.iterations,
            "Vehicles Used": len(self.routes)
        }


class BenchmarkEngine:
    """Runs multiple optimization algorithms on the same VRP problem."""

    def __init__(self):
        self.results = []

    def run_algorithm(self, algorithm_name, optimizer_function):
        start_time = time.time()
        result = optimizer_function()
        runtime = time.time() - start_time

        if not isinstance(result, tuple):
            raise ValueError("Optimizer must return a tuple.")

        if len(result) == 4:
            routes, distance, iterations, history = result
        elif len(result) == 3:
            routes, distance, iterations = result
            history = []
        elif len(result) == 2:
            routes, distance = result
            iterations = 0
            history = []
        else:
            raise ValueError("Optimizer must return 2-4 values.")

        benchmark_result = BenchmarkResult(
            algorithm_name=algorithm_name,
            routes=routes,
            distance=distance,
            runtime=runtime,
            iterations=iterations,
            history=history
        )

        self.results.append(benchmark_result)
        return benchmark_result

    def get_results(self):
        return self.results

    def results_dataframe(self):
        if not self.results:
            return pd.DataFrame(columns=[
                "Algorithm", "Total Distance (km)",
                "Runtime (seconds)", "Iterations", "Vehicles Used"
            ])
        return pd.DataFrame([r.to_dict() for r in self.results])

    def best_result(self):
        if not self.results:
            return None
        return min(self.results, key=lambda r: r.distance)

    def clear(self):
        self.results.clear()


# ============================================================
# ROAD DISTANCE MATRIX
# ============================================================

def build_distance_matrix(G, location_nodes):
    """Build road-distance matrix using Dijkstra. Returns km."""
    n = len(location_nodes)
    distance_matrix = np.full((n, n), 1e9, dtype=float)
    np.fill_diagonal(distance_matrix, 0.0)

    # Directed
    for i, src in enumerate(location_nodes):
        try:
            lengths = nx.single_source_dijkstra_path_length(
                G, src, weight="length"
            )
        except Exception:
            lengths = {}

        for j, tgt in enumerate(location_nodes):
            if tgt in lengths:
                distance_matrix[i, j] = float(lengths[tgt]) / 1000.0

    # Undirected fallback
    if np.any(distance_matrix >= 1e9):
        try:
            uG = G.to_undirected()
            for i, src in enumerate(location_nodes):
                try:
                    lengths = nx.single_source_dijkstra_path_length(
                        uG, src, weight="length"
                    )
                except Exception:
                    lengths = {}
                for j, tgt in enumerate(location_nodes):
                    if distance_matrix[i, j] >= 1e9 and tgt in lengths:
                        distance_matrix[i, j] = float(lengths[tgt]) / 1000.0
        except Exception:
            pass

    return distance_matrix


# ============================================================
# NAVIGATION PATH HELPERS
# ============================================================

def get_safe_navigation_path(G, start_node, end_node):
    """Find actual road path between two snapped OSM nodes."""
    if start_node == end_node:
        return [start_node], "directed"

    # Directed
    try:
        path = nx.shortest_path(G, start_node, end_node, weight="length")
        return path, "directed"
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        pass

    # Undirected fallback
    try:
        uG = G.to_undirected()
        path = nx.shortest_path(uG, start_node, end_node, weight="length")
        return path, "undirected"
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return [], "none"


def get_path_coordinates(G, path):
    coordinates = []
    for node in path:
        try:
            lat = float(G.nodes[node]["y"])
            lon = float(G.nodes[node]["x"])
            coordinates.append([lat, lon])
        except Exception:
            continue
    return coordinates


def get_edge_length(G, u, v, key=None):
    try:
        if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
            data = G.get_edge_data(u, v)
            if data is None:
                return float("inf")
            if key is not None:
                edge_data = data.get(key)
                if edge_data is None:
                    return float("inf")
                return float(edge_data.get("length", 0))
            lengths = []
            for ek, ed in data.items():
                l = ed.get("length")
                if l is not None:
                    lengths.append(float(l))
            if lengths:
                return min(lengths)
        else:
            data = G.get_edge_data(u, v)
            if data is None:
                return float("inf")
            return float(data.get("length", 0))
    except Exception:
        return float("inf")
    return float("inf")


def get_path_distance(G, path):
    if not path or len(path) < 2:
        return 0.0
    total_meters = 0.0
    for i in range(len(path) - 1):
        el = get_edge_length(G, path[i], path[i + 1])
        if math.isfinite(el):
            total_meters += el
    return total_meters / 1000.0


# ============================================================
# ETA HELPERS
# ============================================================

def calculate_eta(distance_km, avg_speed_kmh=30.0):
    if distance_km <= 0:
        return 0.0
    return (distance_km / avg_speed_kmh) * 60.0


def format_eta(minutes):
    minutes = float(minutes)
    if minutes < 60:
        return f"{minutes:.0f} min"
    hours = int(minutes // 60)
    rem = int(round(minutes % 60))
    if rem == 60:
        hours += 1
        rem = 0
    if rem == 0:
        return f"{hours} hr"
    return f"{hours} hr {rem} min"


# ============================================================
# SHARED PROBLEM SETUP
# ============================================================

def prepare_vrp_problem(customers_df, snapped_locations, G,
                        num_vehicles, vehicle_capacity):
    """
    Shared setup for both single-algorithm optimize and benchmark.
    Returns: (location_names, location_nodes, demands,
              customer_indices, distance_matrix, total_demand)
    Raises ValueError on validation failure.
    """
    # Depot
    depot_rows = customers_df[
        customers_df["type"].astype(str).str.lower() == "depot"
    ]
    if len(depot_rows) == 0:
        raise ValueError("No depot found in customer data.")
    depot_id = depot_rows.iloc[0]["id"]

    # Customers
    customer_rows = customers_df[
        customers_df["type"].astype(str).str.lower() != "depot"
    ]
    if len(customer_rows) == 0:
        raise ValueError("No customers found in customer data.")

    # Location names
    location_names = [depot_id]
    for _, row in customer_rows.iterrows():
        location_names.append(row["id"])

    # Location nodes
    location_nodes = []
    missing_snaps = []
    for name in location_names:
        if name not in snapped_locations:
            missing_snaps.append(name)
        else:
            location_nodes.append(snapped_locations[name]["node"])
    if missing_snaps:
        raise ValueError(
            f"Locations not snapped: {', '.join(str(x) for x in missing_snaps)}"
        )

    # Demands
    demands = {}
    for idx, name in enumerate(location_names):
        if idx == 0:
            demands[idx] = 0.0
        else:
            demands[idx] = float(snapped_locations[name]["demand"])

    # Total demand
    total_demand = sum(demands[i] for i in range(1, len(location_names)))

    # Capacity check
    if total_demand > (num_vehicles * vehicle_capacity):
        raise ValueError(
            f"Infeasible: Total demand {total_demand:.1f} kg > "
            f"Total capacity {num_vehicles * vehicle_capacity:.1f} kg"
        )

    # Individual customer check
    oversized = []
    for name in location_names[1:]:
        d = float(snapped_locations[name]["demand"])
        if d > vehicle_capacity:
            oversized.append((name, d))
    if oversized:
        details = ", ".join(f"{n} ({d:.1f} kg)" for n, d in oversized)
        raise ValueError(f"Customer demand exceeds vehicle capacity: {details}")

    # Distance matrix
    distance_matrix = build_distance_matrix(G, location_nodes)
    if np.any(distance_matrix >= 1e9):
        raise ValueError("Some locations are disconnected from the road network.")

    customer_indices = list(range(1, len(location_names)))

    return (location_names, location_nodes, demands,
            customer_indices, distance_matrix, total_demand)


# ============================================================
# PLOTLY THEME
# ============================================================

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#94a3b8", size=12),
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.04)",
        zerolinecolor="rgba(255,255,255,0.04)",
        tickfont=dict(color="#64748b"),
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.04)",
        zerolinecolor="rgba(255,255,255,0.04)",
        tickfont=dict(color="#64748b"),
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor="rgba(255,255,255,0.06)",
        borderwidth=1,
        font=dict(color="#94a3b8", size=11),
    ),
    hoverlabel=dict(
        bgcolor="#1a1a3e",
        bordercolor="rgba(0, 212, 255, 0.3)",
        font=dict(color="#e8ecf4", family="Inter"),
    ),
)


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "graph": None,
    "routes": None,
    "route_stats": None,
    "location_nodes": None,
    "location_names": None,
    "distance_matrix": None,
    "optimization_result": None,
    "benchmark_results": None,
    "convergence_history": None,
    "benchmark_histories": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# ROUTE COLORS
# ============================================================

ROUTE_COLORS = [
    "#00d4ff", "#ff4466", "#00ff88", "#ffaa00",
    "#7c3aed", "#ff6b9d", "#06d6a0", "#ffd166",
    "#118ab2", "#ef476f"
]


# ============================================================
# HEADER
# ============================================================

header_col1, header_col2 = st.columns([5, 1])

with header_col1:
    st.markdown("""
    <div class="hero-container">
        <div class="hero-eyebrow">
            <div class="hero-eyebrow-dot"></div>
            SMART INDIA HACKATHON 2026
        </div>
        <div class="brand-lockup">
            <div class="brand-mark"><span>F</span></div>
            <div>
                <div class="hero-title">FLUXROUTE</div>
                <div class="brand-kicker">Quantum Route Intelligence</div>
            </div>
        </div>
        <div class="hero-subtitle">
            Quantum-Inspired Intelligent Traffic Route Optimization
            in Transportation Systems Using Metaheuristic Optimization
        </div>
        <div class="hero-badges">
            <div class="hero-badge">SIH26137</div>
            <div class="hero-badge hero-badge-purple">EGREEN QUANTA</div>
            <div class="hero-badge hero-badge-green">TRANSPORTATION & LOGISTICS</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with header_col2:
    st.markdown("""
    <div style="padding-top: 36px; text-align: right;">
        <div class="status-badge">
            <div class="status-dot"></div>
            SYSTEM ONLINE
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("""
<div style="text-align: center; padding: 14px 0 18px 0;">
    <div class="brand-lockup" style="justify-content: center;">
        <div class="brand-mark" style="width: 38px; height: 38px; border-radius: 12px 12px 12px 3px;"><span style="font-size: 1.35rem;">F</span></div>
        <div style="text-align: left;">
            <div class="brand-wordmark">FLUXROUTE</div>
            <div class="brand-kicker">QUANTUM ROUTE INTELLIGENCE</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.divider()

# ---- Network Settings ----
st.sidebar.markdown("""
<div class="sidebar-section-title">
    🌍 ROAD NETWORK
</div>
""", unsafe_allow_html=True)

place = st.sidebar.text_input(
    "Map Location",
    value="Koregaon Park, Pune, India",
    label_visibility="collapsed",
    placeholder="Enter city or area..."
)

load_button = st.sidebar.button(
    "🌐 Load Road Network",
    use_container_width=True,
    type="primary"
)

# ---- Load Road Network ----
if load_button:
    with st.spinner("Loading OpenStreetMap road network..."):
        try:
            G = load_city_graph(place)
            validate_graph(G)
            st.session_state["graph"] = G
            for k in ["routes", "route_stats", "location_nodes",
                       "location_names", "distance_matrix",
                       "optimization_result", "benchmark_results",
                       "convergence_history", "benchmark_histories"]:
                st.session_state[k] = None
            st.toast("✅ Road network loaded!", icon="🌐")
        except Exception as e:
            st.error(f"❌ Unable to load road network: {e}")

st.sidebar.divider()

# ---- Customer Data ----
st.sidebar.markdown("""
<div class="sidebar-section-title">
    📦 ORDER DATA
</div>
""", unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader(
    "Upload customer CSV",
    type=["csv"],
    label_visibility="collapsed"
)

if uploaded_file is not None:
    customers_df = pd.read_csv(uploaded_file)
else:
    try:
        customers_df = pd.read_csv(
            os.path.join(PROJECT_ROOT, "data", "customers.csv")
        )
    except FileNotFoundError:
        st.error("data/customers.csv not found.")
        st.stop()

# ---- Customer Validation ----
required_columns = ["id", "type", "latitude", "longitude", "demand"]
missing = [c for c in required_columns if c not in customers_df.columns]
if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()

# ---- Normalize ----
customers_df["type"] = customers_df["type"].astype(str).str.strip()
customers_df["latitude"] = pd.to_numeric(customers_df["latitude"], errors="coerce")
customers_df["longitude"] = pd.to_numeric(customers_df["longitude"], errors="coerce")
customers_df["demand"] = pd.to_numeric(customers_df["demand"], errors="coerce")

if customers_df[["latitude", "longitude", "demand"]].isnull().any().any():
    st.error("Customer CSV contains invalid latitude, longitude or demand values.")
    st.stop()

# ---- Order Statistics ----
customer_mask = customers_df["type"].str.lower() != "depot"
depot_mask = customers_df["type"].str.lower() == "depot"
customer_count = len(customers_df[customer_mask])
depot_count = len(customers_df[depot_mask])
total_demand = customers_df[customer_mask]["demand"].sum()

st.sidebar.divider()

# ---- Sidebar Stats ----
st.sidebar.markdown(f"""
<div class="sidebar-section">
    <div class="sidebar-section-title">📊 ORDER SUMMARY</div>
    <div class="sidebar-stat">
        <span class="sidebar-stat-label">Customers</span>
        <span class="sidebar-stat-value">{customer_count}</span>
    </div>
    <div class="sidebar-stat">
        <span class="sidebar-stat-label">Depots</span>
        <span class="sidebar-stat-value">{depot_count}</span>
    </div>
    <div class="sidebar-stat">
        <span class="sidebar-stat-label">Total Demand</span>
        <span class="sidebar-stat-value">{total_demand:.0f} kg</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---- Location dictionary ----
locations = {}
for _, row in customers_df.iterrows():
    locations[row["id"]] = {
        "type": row["type"],
        "latitude": float(row["latitude"]),
        "longitude": float(row["longitude"]),
        "demand": float(row["demand"])
    }


# ============================================================
# CHECK NETWORK LOADED
# ============================================================

if st.session_state["graph"] is None:
    st.markdown("""
    <div class="glass-card landing-container">
        <div class="landing-icon">🌐</div>
        <div class="landing-title">No Road Network Loaded</div>
        <div class="landing-desc">
            Load a road network from OpenStreetMap to begin quantum-inspired route optimization.
            Enter a city or area in the sidebar and click <b>Load Road Network</b>.
        </div>
        <div class="landing-steps">
            <div class="landing-step">
                <div class="landing-step-num">01</div>
                <div class="landing-step-text">Enter location in sidebar</div>
            </div>
            <div class="landing-step">
                <div class="landing-step-num">02</div>
                <div class="landing-step-text">Load road network</div>
            </div>
            <div class="landing-step">
                <div class="landing-step-num">03</div>
                <div class="landing-step-text">Optimize routes</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

G = st.session_state["graph"]

# ---- Graph Info ----
try:
    info = describe_graph(G)
except Exception:
    info = {"nodes": len(G.nodes), "edges": len(G.edges)}


# ============================================================
# TOP METRICS ROW
# ============================================================

st.markdown("""
<div class="section-header">
    <div class="section-icon">📡</div>
    <div class="section-title">Network Overview</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-subtitle">Real-time road network telemetry from OpenStreetMap</div>',
            unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-icon">🔗</div>
        <div class="metric-value">{info['nodes']:,}</div>
        <div class="metric-label">Road Nodes</div>
        <div class="metric-delta">Intersections & endpoints</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div class="metric-card purple">
        <div class="metric-icon">🛤️</div>
        <div class="metric-value">{info['edges']:,}</div>
        <div class="metric-label">Road Segments</div>
        <div class="metric-delta">Directed edges</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown(f"""
    <div class="metric-card green">
        <div class="metric-icon">📦</div>
        <div class="metric-value">{customer_count}</div>
        <div class="metric-label">Delivery Points</div>
        <div class="metric-delta">{depot_count} depot(s) loaded</div>
    </div>
    """, unsafe_allow_html=True)

with m4:
    st.markdown(f"""
    <div class="metric-card amber">
        <div class="metric-icon">⚖️</div>
        <div class="metric-value">{total_demand:.0f} kg</div>
        <div class="metric-label">Total Demand</div>
        <div class="metric-delta">Across all customers</div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# SNAP LOCATIONS
# ============================================================

with st.spinner("Snapping orders to road network..."):
    try:
        snapped_locations = snap_locations(G, locations)
    except Exception as e:
        st.error(f"Unable to snap locations: {e}")
        st.stop()

unsnapped = [lid for lid in locations if lid not in snapped_locations]
if unsnapped:
    st.warning(f"⚠️ Some locations could not be snapped: {', '.join(str(x) for x in unsnapped)}")

# Sidebar network info
st.sidebar.markdown(f"""
<div class="sidebar-section">
    <div class="sidebar-section-title">🌐 NETWORK STATUS</div>
    <div class="sidebar-stat">
        <span class="sidebar-stat-label">Location</span>
        <span class="sidebar-stat-value" style="font-size: 0.72rem;">{place[:25]}</span>
    </div>
    <div class="sidebar-stat">
        <span class="sidebar-stat-label">Road Nodes</span>
        <span class="sidebar-stat-value">{info['nodes']:,}</span>
    </div>
    <div class="sidebar-stat">
        <span class="sidebar-stat-label">Road Edges</span>
        <span class="sidebar-stat-value">{info['edges']:,}</span>
    </div>
    <div class="sidebar-stat">
        <span class="sidebar-stat-label">Snapped</span>
        <span class="sidebar-stat-value" style="color: #00ff88;">{len(snapped_locations)} ✓</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# MAP
# ============================================================

st.markdown("""
<div class="section-header">
    <div class="section-icon">🗺️</div>
    <div class="section-title">Live Network Map</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-subtitle">Interactive map with real road topology and optimized route overlay</div>',
            unsafe_allow_html=True)

# Center
first_node = list(G.nodes)[0]
center_lat = float(G.nodes[first_node]["y"])
center_lon = float(G.nodes[first_node]["x"])

# Create map with dark theme default
m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=14,
    tiles=None,
    control_scale=True
)

# Tile layers — dark first
folium.TileLayer(
    "CartoDB dark_matter",
    name="🌑 Dark Map"
).add_to(m)

folium.TileLayer(
    "CartoDB positron",
    name="🌕 Light Map"
).add_to(m)

folium.TileLayer(
    "OpenStreetMap",
    name="🗺️ Street Map"
).add_to(m)

# Road network overlay
try:
    edges = ox.graph_to_gdfs(G, nodes=False)
    folium.GeoJson(
        edges.to_json(),
        name="Road Network",
        style_function=lambda f: {
            "color": "#1e293b",
            "weight": 1.2,
            "opacity": 0.35
        }
    ).add_to(m)
except Exception as e:
    st.warning(f"Unable to draw road network: {e}")

# ---- Draw optimized routes ----
if st.session_state["routes"] is not None:
    routes = st.session_state["routes"]
    loc_nodes = st.session_state["location_nodes"]
    loc_names = st.session_state["location_names"]

    for v_num, route in enumerate(routes):
        color = ROUTE_COLORS[v_num % len(ROUTE_COLORS)]
        complete = [0] + list(route) + [0]

        for i in range(len(complete) - 1):
            si, ei = complete[i], complete[i + 1]
            sn, en = loc_nodes[si], loc_nodes[ei]

            nav_path, path_type = get_safe_navigation_path(G, sn, en)
            if not nav_path:
                continue

            coords = get_path_coordinates(G, nav_path)
            if not coords:
                continue

            rd = get_path_distance(G, nav_path)
            plabel = {"directed": "OSM directed", "undirected": "OSM fallback"}.get(
                path_type, "No road"
            )

            folium.PolyLine(
                locations=coords,
                color=color,
                weight=5,
                opacity=0.85,
                dash_array="",
                tooltip=f"🚚 Vehicle {v_num + 1}",
                popup=folium.Popup(
                    f"""<div style="width:240px;font-family:Inter,Arial,sans-serif;
                    background:#0a0a1e;color:#e2e8f0;padding:14px;border-radius:12px;
                    border:1px solid rgba(0,212,255,0.15);">
                    <h4 style="color:{color};margin:0 0 10px 0;font-size:14px;">
                    🚚 Vehicle {v_num + 1}</h4>
                    <div style="font-size:12px;line-height:1.8;">
                    <b>From:</b> {loc_names[si]}<br>
                    <b>To:</b> {loc_names[ei]}<br>
                    <b>Distance:</b> {rd:.2f} km<br>
                    <b>Routing:</b> {plabel}
                    </div>
                    </div>""",
                    max_width=300
                )
            ).add_to(m)

# ---- Depot & Customer Markers ----
c_num = 1
for loc_id, data in snapped_locations.items():
    lat, lon = data["latitude"], data["longitude"]

    if str(data["type"]).lower() == "depot":
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(
                f"""<div style="width:200px;font-family:Inter,Arial;
                background:#0a0a1e;color:#e2e8f0;padding:14px;border-radius:12px;
                border:1px solid rgba(255,68,102,0.2);">
                <h4 style="color:#ff4466;margin:0 0 8px;">🏭 Depot</h4>
                <div style="font-size:12px;line-height:1.6;">
                <b>ID:</b> {loc_id}<br>
                <b>Node:</b> {data['node']}
                </div>
                </div>""",
                max_width=250
            ),
            tooltip="🏭 Depot",
            icon=folium.Icon(icon="home", prefix="fa", color="red")
        ).add_to(m)
    else:
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(
                f"""<div style="width:210px;font-family:Inter,Arial;
                background:#0a0a1e;color:#e2e8f0;padding:14px;border-radius:12px;
                border:1px solid rgba(0,212,255,0.15);">
                <h4 style="color:#00d4ff;margin:0 0 8px;">📦 Customer {c_num}</h4>
                <div style="font-size:12px;line-height:1.6;">
                <b>ID:</b> {loc_id}<br>
                <b>Demand:</b> {data['demand']:.1f} kg<br>
                <b>Node:</b> {data['node']}
                </div>
                </div>""",
                max_width=250
            ),
            tooltip=f"📦 Customer {c_num} ({data['demand']:.0f} kg)",
            icon=folium.DivIcon(
                html=f"""<div style="
                    background: linear-gradient(135deg, #00d4ff, #7c3aed);
                    color:white;border-radius:50%;
                    width:30px;height:30px;
                    text-align:center;line-height:30px;
                    font-weight:bold;font-family:Inter,Arial;
                    font-size:12px;
                    border:2px solid rgba(255,255,255,0.9);
                    box-shadow:0 2px 12px rgba(0,212,255,0.5);
                ">{c_num}</div>""",
                icon_size=(30, 30),
                icon_anchor=(15, 15)
            )
        ).add_to(m)
        c_num += 1

# ---- Legend ----
legend_html = """
<div style="
    position:fixed; bottom:25px; left:25px; z-index:9999;
    background:rgba(5,5,16,0.92); backdrop-filter:blur(16px);
    -webkit-backdrop-filter:blur(16px);
    padding:16px 20px; border-radius:14px;
    border:1px solid rgba(255,255,255,0.08);
    box-shadow:0 4px 24px rgba(0,0,0,0.5);
    font-family:Inter,Arial; font-size:12px; color:#e2e8f0;
">
<b style="font-size:13px; letter-spacing:0.5px;">🗺️ Map Legend</b><br><br>
<span style="display:inline-block;width:14px;height:14px;
    background:linear-gradient(135deg,#00d4ff,#7c3aed);
    border-radius:50%;vertical-align:middle;
    box-shadow:0 0 6px rgba(0,212,255,0.3);"></span>
    <span style="vertical-align:middle;"> Customer</span><br><br>
<span style="display:inline-block;width:14px;height:14px;
    background:#ff4466;border-radius:50%;vertical-align:middle;
    box-shadow:0 0 6px rgba(255,68,102,0.3);"></span>
    <span style="vertical-align:middle;"> Depot</span><br><br>
<span style="display:inline-block;width:28px;height:4px;
    background:linear-gradient(90deg,#00d4ff,#7c3aed);
    border-radius:3px;vertical-align:middle;"></span>
    <span style="vertical-align:middle;"> Optimized Route</span><br><br>
<span style="display:inline-block;width:28px;height:2px;
    background:#1e293b;vertical-align:middle;"></span>
    <span style="vertical-align:middle;"> Road Network</span>
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

folium.LayerControl(position="topright").add_to(m)

st_folium(m, width=None, height=600, returned_objects=["last_object_clicked"])


# ============================================================
# ROUTE PLANNING
# ============================================================

st.markdown("""
<div class="section-header">
    <div class="section-icon">🚀</div>
    <div class="section-title">Route Optimization</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-subtitle">Configure the optimization algorithm and vehicle fleet parameters</div>',
            unsafe_allow_html=True)

rc1, rc2, rc3 = st.columns(3)

with rc1:
    algorithm = st.selectbox(
        "Optimization Algorithm",
        [
            "QPSO — Quantum-Inspired PSO",
            "PSO — Particle Swarm Optimization",
            "GA — Genetic Algorithm"
        ]
    )

with rc2:
    num_vehicles = st.number_input(
        "Number of Vehicles",
        min_value=1, max_value=100, value=2, step=1
    )

with rc3:
    vehicle_capacity = st.number_input(
        "Vehicle Capacity (kg)",
        min_value=1, max_value=10000, value=200, step=1
    )

traffic_enabled = st.checkbox(
    "Apply dynamic traffic conditions",
    value=True,
    help="Adjust road costs using time of day, weather, and simulated congestion."
)

traffic_col1, traffic_col2, traffic_col3 = st.columns(3)
with traffic_col1:
    traffic_hour = st.slider("Traffic hour", 0, 23, 8)
with traffic_col2:
    traffic_weather = st.selectbox(
        "Weather", ["clear", "rain", "heavy_rain", "fog"]
    )
with traffic_col3:
    congestion_probability = st.slider(
        "Congestion probability", 0.0, 1.0, 0.15, 0.05
    )

# ---- Default parameters ----
num_particles = 30
max_iterations = 100
beta = 0.75
inertia = 0.7
cognitive = 1.5
social = 1.5
population_size = 30
mutation_rate = 0.1

# ---- Algorithm Configuration ----
if algorithm == "QPSO — Quantum-Inspired PSO":
    with st.expander("⚛️ QPSO Configuration", expanded=True):
        qc1, qc2, qc3 = st.columns(3)
        with qc1:
            num_particles = st.number_input("Particles", 5, 200, 30, 5)
        with qc2:
            max_iterations = st.number_input("Iterations", 10, 1000, 100, 10)
        with qc3:
            beta = st.slider("Contraction-Expansion β", 0.1, 1.5, 0.75, 0.05)

elif algorithm == "PSO — Particle Swarm Optimization":
    with st.expander("🌀 PSO Configuration", expanded=True):
        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            num_particles = st.number_input("Particles", 5, 200, 30, 5)
        with pc2:
            max_iterations = st.number_input("Iterations", 10, 1000, 100, 10)
        with pc3:
            inertia = st.slider("Inertia", 0.1, 1.5, 0.7, 0.05)
        cognitive = st.slider("Cognitive coefficient", 0.1, 3.0, 1.5, 0.1)
        social = st.slider("Social coefficient", 0.1, 3.0, 1.5, 0.1)

else:
    with st.expander("🧬 Genetic Algorithm Configuration", expanded=True):
        gc1, gc2, gc3 = st.columns(3)
        with gc1:
            population_size = st.number_input("Population Size", 5, 200, 30, 5)
        with gc2:
            max_iterations = st.number_input("Iterations", 10, 1000, 100, 10)
        with gc3:
            mutation_rate = st.slider("Mutation Rate", 0.01, 1.0, 0.1, 0.01)

# ---- Action Buttons ----
btn1, btn2 = st.columns([5, 1])

with btn1:
    optimize_button = st.button(
        "🚀 Generate Optimized Routes",
        use_container_width=True, type="primary"
    )

with btn2:
    clear_button = st.button("🗑️ Clear", use_container_width=True)

if clear_button:
    for k in ["routes", "route_stats", "optimization_result",
              "benchmark_results", "convergence_history", "benchmark_histories"]:
        st.session_state[k] = None
    st.rerun()


# ============================================================
# OPTIMIZATION
# ============================================================

if optimize_button:
    try:
        (location_names, location_nodes, demands,
         customer_indices, distance_matrix, td) = prepare_vrp_problem(
            customers_df, snapped_locations, G,
            num_vehicles, vehicle_capacity
        )

        if traffic_enabled:
            distance_matrix, traffic_info = DynamicWeights(
                distance_matrix, seed=42
            ).get_adjusted_matrix(
                hour=traffic_hour,
                weather=traffic_weather,
                congestion_probability=congestion_probability
            )
            st.info(
                f"Traffic model: {traffic_info['hour']:02d}:00, "
                f"{traffic_weather}, {traffic_info['congested_segments']} "
                "congested segments"
            )

        progress = st.progress(0, text="⚡ Preparing road network...")

        start_time = time.time()
        progress.progress(25, text="📊 Road distance matrix ready.")

        # Create optimizer
        if algorithm == "QPSO — Quantum-Inspired PSO":
            optimizer = QPSO_VRP(
                distance_matrix=distance_matrix,
                demands=demands,
                customer_indices=customer_indices,
                depot_index=0,
                vehicle_capacity=int(vehicle_capacity),
                num_vehicles=int(num_vehicles),
                num_particles=int(num_particles),
                max_iterations=int(max_iterations),
                beta=float(beta),
                seed=42
            )
        elif algorithm == "PSO — Particle Swarm Optimization":
            optimizer = PSO_VRP(
                distance_matrix=distance_matrix,
                demands=demands,
                customer_indices=customer_indices,
                depot_index=0,
                vehicle_capacity=int(vehicle_capacity),
                num_vehicles=int(num_vehicles),
                num_particles=int(num_particles),
                max_iterations=int(max_iterations),
                inertia=float(inertia),
                cognitive=float(cognitive),
                social=float(social),
                seed=42
            )
        else:
            optimizer = GeneticAlgorithmVRP(
                distance_matrix=distance_matrix,
                demands=demands,
                customer_indices=customer_indices,
                depot_index=0,
                vehicle_capacity=int(vehicle_capacity),
                num_vehicles=int(num_vehicles),
                population_size=int(population_size),
                max_iterations=int(max_iterations),
                mutation_rate=float(mutation_rate),
                seed=42
            )

        progress.progress(40, text=f"⚛️ Running {algorithm.split(' — ')[0]}...")

        with st.spinner(f"🚀 {algorithm.split(' — ')[0]} is optimizing..."):
            result = optimizer.optimize()

        # Support 2, 3, or 4 return values
        if not isinstance(result, tuple):
            raise ValueError("Optimizer.optimize() must return a tuple.")

        if len(result) == 4:
            routes, best_distance, iterations, history = result
        elif len(result) == 3:
            routes, best_distance, iterations = result
            history = []
        elif len(result) == 2:
            routes, best_distance = result
            iterations = max_iterations
            history = []
        else:
            raise ValueError("Optimizer must return 2-4 values.")

        progress.progress(80, text="📈 Calculating route statistics...")

        if routes is None:
            st.error("❌ Optimizer did not generate routes.")
            st.stop()

        if len(routes) > num_vehicles:
            st.error("❌ Optimizer generated more routes than available vehicles.")
            st.stop()

        # Save results
        st.session_state["routes"] = routes
        st.session_state["location_nodes"] = location_nodes
        st.session_state["location_names"] = location_names
        st.session_state["distance_matrix"] = distance_matrix
        st.session_state["convergence_history"] = history

        runtime = time.time() - start_time

        st.session_state["optimization_result"] = {
            "algorithm": algorithm,
            "distance": float(best_distance),
            "runtime": runtime,
            "iterations": int(iterations)
        }

        # Route statistics
        route_stats = []
        for v_num, route in enumerate(routes):
            route = list(route)
            load = sum(demands.get(int(c), 0.0) for c in route)

            try:
                dist = float(optimizer.route_distance(route))
            except Exception:
                cr = [0] + route + [0]
                dist = sum(
                    distance_matrix[cr[i], cr[i + 1]]
                    for i in range(len(cr) - 1)
                )

            names = [str(location_names[int(c)]) for c in route]
            eta_min = calculate_eta(dist)
            eta_txt = format_eta(eta_min)
            util = (load / vehicle_capacity * 100) if vehicle_capacity > 0 else 0.0

            route_stats.append({
                "Vehicle": f"Vehicle {v_num + 1}",
                "Customers": len(route),
                "Load (kg)": round(load, 2),
                "Capacity (kg)": vehicle_capacity,
                "Utilization (%)": round(util, 1),
                "Distance (km)": round(dist, 2),
                "ETA": eta_txt,
                "Route": "DEPOT → " + " → ".join(names) + " → DEPOT",
                "Color": ROUTE_COLORS[v_num % len(ROUTE_COLORS)]
            })

        st.session_state["route_stats"] = pd.DataFrame(route_stats)

        progress.progress(100, text="✅ Optimization completed!")

        st.toast(
            f"✅ {algorithm.split(' — ')[0]} completed in {runtime:.2f}s — "
            f"Total distance: {best_distance:.2f} km",
            icon="🚀"
        )

    except Exception as e:
        st.error(f"❌ Route optimization failed: {e}")


# ============================================================
# OPTIMIZATION RESULTS
# ============================================================

if st.session_state["route_stats"] is not None:
    stats_df = st.session_state["route_stats"]
    result_info = st.session_state["optimization_result"]

    st.markdown("""
    <div class="section-header">
        <div class="section-icon">📊</div>
        <div class="section-title">Optimization Dashboard</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-subtitle">Performance metrics and route analysis from the optimization engine</div>',
                unsafe_allow_html=True)

    # ---- KPIs ----
    total_distance = stats_df["Distance (km)"].sum()
    total_load = stats_df["Load (kg)"].sum()
    avg_util = stats_df["Utilization (%)"].mean() if len(stats_df) > 0 else 0.0

    k1, k2, k3, k4, k5, k6 = st.columns(6)

    algo_short = result_info['algorithm'].split(' — ')[0]

    with k1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">⚛️</div>
            <div class="metric-value metric-value-small">{algo_short}</div>
            <div class="metric-label">Algorithm</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">🛣️</div>
            <div class="metric-value">{total_distance:.2f}</div>
            <div class="metric-label">Total KM</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="metric-card purple">
            <div class="metric-icon">🚚</div>
            <div class="metric-value">{len(stats_df)}</div>
            <div class="metric-label">Vehicles</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="metric-card green">
            <div class="metric-icon">📊</div>
            <div class="metric-value">{avg_util:.1f}%</div>
            <div class="metric-label">Avg Util</div>
        </div>
        """, unsafe_allow_html=True)
    with k5:
        st.markdown(f"""
        <div class="metric-card amber">
            <div class="metric-icon">⏱️</div>
            <div class="metric-value">{result_info['runtime']:.2f}s</div>
            <div class="metric-label">Runtime</div>
        </div>
        """, unsafe_allow_html=True)
    with k6:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">🔄</div>
            <div class="metric-value">{result_info['iterations']}</div>
            <div class="metric-label">Iterations</div>
        </div>
        """, unsafe_allow_html=True)

    # ---- TABS for results ----
    tab_convergence, tab_vehicles, tab_routes, tab_export = st.tabs([
        "📈 Convergence", "🚚 Vehicle Performance", "🛣️ Route Details", "📥 Export"
    ])

    # ---- Convergence Chart (Plotly) ----
    with tab_convergence:
        history = st.session_state.get("convergence_history")
        if history and len(history) > 1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(len(history))),
                y=history,
                mode='lines',
                name=algo_short,
                line=dict(
                    color='#00d4ff',
                    width=2.5,
                    shape='spline',
                    smoothing=0.8
                ),
                fill='tozeroy',
                fillcolor='rgba(0, 212, 255, 0.05)',
                hovertemplate='<b>Iteration %{x}</b><br>Distance: %{y:.3f} km<extra></extra>'
            ))

            # Add improvement markers
            if len(history) > 2:
                improvements_x = []
                improvements_y = []
                for i in range(1, len(history)):
                    if history[i] < history[i-1]:
                        improvements_x.append(i)
                        improvements_y.append(history[i])
                # Only show first and last improvement for cleanliness
                if improvements_x:
                    fig.add_trace(go.Scatter(
                        x=[improvements_x[0], improvements_x[-1]],
                        y=[improvements_y[0], improvements_y[-1]],
                        mode='markers',
                        name='Key Improvements',
                        marker=dict(
                            color='#00ff88',
                            size=8,
                            symbol='diamond',
                            line=dict(color='rgba(0,255,136,0.5)', width=2)
                        ),
                        hovertemplate='<b>Improvement at iteration %{x}</b><br>Distance: %{y:.3f} km<extra></extra>'
                    ))

            fig.update_layout(
                **PLOTLY_LAYOUT,
                title=dict(
                    text=f"<b>{algo_short} Convergence</b>",
                    font=dict(color="#e8ecf4", size=16),
                    x=0.02
                ),
                xaxis_title="Iteration",
                yaxis_title="Best Distance (km)",
                height=400,
                showlegend=True,
            )

            st.plotly_chart(fig, use_container_width=True)

            # Improvement summary
            if len(history) > 1:
                initial = history[0]
                final = history[-1]
                improvement = ((initial - final) / initial * 100) if initial > 0 else 0
                ic1, ic2, ic3 = st.columns(3)
                with ic1:
                    st.metric("Initial Distance", f"{initial:.3f} km")
                with ic2:
                    st.metric("Final Distance", f"{final:.3f} km")
                with ic3:
                    st.metric("Improvement", f"{improvement:.1f}%")
        else:
            st.info("No convergence history available for this run.")

    # ---- Vehicle Performance ----
    with tab_vehicles:
        # Data table
        display_cols = [c for c in stats_df.columns if c != "Color"]
        st.dataframe(
            stats_df[display_cols],
            use_container_width=True,
            hide_index=True
        )

        st.markdown("##### 📦 Capacity Utilization")

        for _, row in stats_df.iterrows():
            color = row.get("Color", "#00d4ff")
            util_pct = row["Utilization (%)"]

            # Color based on utilization
            if util_pct >= 80:
                bar_color = "#00ff88"
            elif util_pct >= 50:
                bar_color = "#ffaa00"
            else:
                bar_color = "#ff4466"

            st.markdown(f"""
            <div style="margin-bottom: 14px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <span style="font-size: 0.85rem; font-weight: 700; color: {color};">
                        🚚 {row['Vehicle']}
                    </span>
                    <span style="font-size: 0.8rem; color: var(--text-secondary);
                          font-family: 'JetBrains Mono', monospace;">
                        {row['Load (kg)']:.0f} / {row['Capacity (kg)']:.0f} kg
                        <span style="color: {bar_color}; font-weight: 700;">
                            ({util_pct:.1f}%)
                        </span>
                    </span>
                </div>
                <div class="util-bar-container">
                    <div class="util-bar-fill" style="
                        width: {min(util_pct, 100)}%;
                        background: linear-gradient(90deg, {color}, {bar_color});
                    "></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Vehicle distance chart (Plotly)
        if len(stats_df) > 1:
            fig_bar = go.Figure()
            for i, row in stats_df.iterrows():
                color = row.get("Color", "#00d4ff")
                fig_bar.add_trace(go.Bar(
                    x=[row["Vehicle"]],
                    y=[row["Distance (km)"]],
                    name=row["Vehicle"],
                    marker=dict(
                        color=color,
                        line=dict(color='rgba(255,255,255,0.1)', width=1),
                        opacity=0.85,
                    ),
                    text=f'{row["Distance (km)"]:.2f} km',
                    textposition='outside',
                    textfont=dict(color='#94a3b8', size=11),
                    hovertemplate=f'<b>{row["Vehicle"]}</b><br>'
                                  f'Distance: {row["Distance (km)"]:.2f} km<br>'
                                  f'Load: {row["Load (kg)"]:.1f} kg<br>'
                                  f'ETA: {row["ETA"]}<extra></extra>'
                ))

            fig_bar.update_layout(
                **PLOTLY_LAYOUT,
                title=dict(
                    text="<b>Distance per Vehicle</b>",
                    font=dict(color="#e8ecf4", size=14),
                    x=0.02
                ),
                yaxis_title="Distance (km)",
                height=350,
                showlegend=False,
                bargap=0.3,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    # ---- Route Cards ----
    with tab_routes:
        for _, row in stats_df.iterrows():
            color = row.get("Color", "#00d4ff")
            v_idx = int(row["Vehicle"].split()[-1])

            # Build styled route path
            route_parts = row["Route"].split(" → ")
            styled_parts = []
            for part in route_parts:
                if part == "DEPOT":
                    styled_parts.append(f'<span class="depot">DEPOT</span>')
                else:
                    styled_parts.append(f'<span class="customer">{part}</span>')
            styled_route = ' <span class="arrow">→</span> '.join(styled_parts)

            st.markdown(f"""
            <div class="route-card" style="border-left-color: {color};">
                <div class="route-header">
                    <div class="route-title">
                        <div class="route-vehicle-badge" style="background: {color};">
                            {v_idx}
                        </div>
                        Vehicle {v_idx}
                    </div>
                    <span style="font-size: 0.78rem; color: {color};
                          font-weight: 700; font-family: 'JetBrains Mono', monospace;">
                        {row['Distance (km)']:.2f} km
                    </span>
                </div>
                <div class="route-stats">
                    <div class="route-stat">
                        <div class="route-stat-value">{row['Customers']}</div>
                        <div class="route-stat-label">Stops</div>
                    </div>
                    <div class="route-stat">
                        <div class="route-stat-value">{row['Load (kg)']:.0f} kg</div>
                        <div class="route-stat-label">Load</div>
                    </div>
                    <div class="route-stat">
                        <div class="route-stat-value">{row['Utilization (%)']:.1f}%</div>
                        <div class="route-stat-label">Capacity</div>
                    </div>
                    <div class="route-stat">
                        <div class="route-stat-value">{row['ETA']}</div>
                        <div class="route-stat-label">ETA</div>
                    </div>
                </div>
                <div class="route-path">{styled_route}</div>
            </div>
            """, unsafe_allow_html=True)

    # ---- Export ----
    with tab_export:
        st.markdown("##### Download your optimization results")

        dl1, dl2 = st.columns(2)

        with dl1:
            export_df = stats_df[[c for c in stats_df.columns if c != "Color"]]
            st.download_button(
                "⬇️ Route Results (CSV)",
                data=export_df.to_csv(index=False),
                file_name="fluxroute_routes.csv",
                mime="text/csv",
                use_container_width=True
            )

        with dl2:
            summary_df = pd.DataFrame({
                "Metric": [
                    "Algorithm", "Total Distance (km)", "Vehicles Used",
                    "Runtime (seconds)", "Iterations", "Avg Utilization (%)"
                ],
                "Value": [
                    result_info["algorithm"],
                    round(total_distance, 3),
                    len(stats_df),
                    round(result_info["runtime"], 4),
                    result_info["iterations"],
                    round(avg_util, 2)
                ]
            })
            st.download_button(
                "⬇️ Optimization Summary (CSV)",
                data=summary_df.to_csv(index=False),
                file_name="fluxroute_summary.csv",
                mime="text/csv",
                use_container_width=True
            )


# ============================================================
# ALGORITHM BENCHMARK
# ============================================================

st.markdown("""
<div class="section-header">
    <div class="section-icon">⚖️</div>
    <div class="section-title">Algorithm Benchmark</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-subtitle">Compare seven route optimizers on the same traffic-aware VRP problem</div>',
            unsafe_allow_html=True)

benchmark_button = st.button(
    "⚖️ Run Algorithm Benchmark",
    use_container_width=True, type="primary"
)

if benchmark_button:
    try:
        (location_names, location_nodes, demands,
         customer_indices, distance_matrix, td) = prepare_vrp_problem(
            customers_df, snapped_locations, G,
            num_vehicles, vehicle_capacity
        )

        if traffic_enabled:
            distance_matrix, traffic_info = DynamicWeights(
                distance_matrix, seed=42
            ).get_adjusted_matrix(
                hour=traffic_hour,
                weather=traffic_weather,
                congestion_probability=congestion_probability
            )

        def run_qpso():
            opt = QPSO_VRP(
                distance_matrix=distance_matrix, demands=demands,
                customer_indices=customer_indices, depot_index=0,
                vehicle_capacity=int(vehicle_capacity),
                num_vehicles=int(num_vehicles),
                num_particles=int(num_particles),
                max_iterations=int(max_iterations),
                beta=float(beta), seed=42
            )
            return opt.optimize()

        def run_pso():
            opt = PSO_VRP(
                distance_matrix=distance_matrix, demands=demands,
                customer_indices=customer_indices, depot_index=0,
                vehicle_capacity=int(vehicle_capacity),
                num_vehicles=int(num_vehicles),
                num_particles=int(num_particles),
                max_iterations=int(max_iterations),
                inertia=float(inertia),
                cognitive=float(cognitive),
                social=float(social), seed=42
            )
            return opt.optimize()

        def run_ga():
            opt = GeneticAlgorithmVRP(
                distance_matrix=distance_matrix, demands=demands,
                customer_indices=customer_indices, depot_index=0,
                vehicle_capacity=int(vehicle_capacity),
                num_vehicles=int(num_vehicles),
                population_size=int(population_size),
                max_iterations=int(max_iterations),
                mutation_rate=float(mutation_rate), seed=42
            )
            return opt.optimize()

        def run_aco():
            opt = ACOVRP(
                distance_matrix=distance_matrix, demands=demands,
                customer_indices=customer_indices, depot_index=0,
                vehicle_capacity=int(vehicle_capacity),
                num_vehicles=int(num_vehicles), num_ants=int(num_particles),
                max_iterations=int(max_iterations), seed=42
            )
            return opt.optimize()

        def run_sa():
            opt = SimulatedAnnealingVRP(
                distance_matrix=distance_matrix, demands=demands,
                customer_indices=customer_indices, depot_index=0,
                vehicle_capacity=int(vehicle_capacity),
                num_vehicles=int(num_vehicles),
                max_iterations=int(max_iterations), seed=42
            )
            return opt.optimize()

        def run_tabu():
            opt = TabuSearchVRP(
                distance_matrix=distance_matrix, demands=demands,
                customer_indices=customer_indices, depot_index=0,
                vehicle_capacity=int(vehicle_capacity),
                num_vehicles=int(num_vehicles),
                max_iterations=int(max_iterations), seed=42
            )
            return opt.optimize()

        def run_alns():
            opt = ALNS_VRP(
                distance_matrix=distance_matrix, demands=demands,
                customer_indices=customer_indices, depot_index=0,
                vehicle_capacity=int(vehicle_capacity),
                num_vehicles=int(num_vehicles),
                max_iterations=int(max_iterations), seed=42
            )
            return opt.optimize()

        bm = BenchmarkEngine()

        progress = st.progress(0, text="⚡ Starting benchmark...")

        progress.progress(10, text="⚛️ Running QPSO...")
        bm.run_algorithm("QPSO", run_qpso)

        progress.progress(40, text="🌀 Running PSO...")
        bm.run_algorithm("PSO", run_pso)

        progress.progress(70, text="🧬 Running Genetic Algorithm...")
        bm.run_algorithm("GA", run_ga)

        progress.progress(78, text="🐜 Running ACO...")
        bm.run_algorithm("ACO", run_aco)

        progress.progress(84, text="🌡️ Running Simulated Annealing...")
        bm.run_algorithm("SA", run_sa)

        progress.progress(92, text="Running Tabu Search...")
        bm.run_algorithm("Tabu Search", run_tabu)

        progress.progress(96, text="🧩 Running ALNS...")
        bm.run_algorithm("ALNS", run_alns)

        progress.progress(100, text="✅ Benchmark completed!")

        benchmark_df = bm.results_dataframe()
        best = bm.best_result()

        if best is not None:
            st.session_state["benchmark_results"] = benchmark_df

            # Store convergence histories for comparison chart
            bm_histories = {}
            for r in bm.get_results():
                if r.history:
                    bm_histories[r.algorithm_name] = r.history
            st.session_state["benchmark_histories"] = bm_histories

            st.toast(
                f"🏆 Best: {best.algorithm_name} — {best.distance:.2f} km in {best.runtime:.3f}s",
                icon="🏆"
            )
        else:
            st.error("❌ Benchmark did not produce results.")

    except Exception as e:
        st.error(f"❌ Benchmark failed: {e}")


# ============================================================
# DISPLAY BENCHMARK RESULTS
# ============================================================

if st.session_state.get("benchmark_results") is not None:
    benchmark_df = st.session_state["benchmark_results"]

    # Winner highlight
    valid = benchmark_df[benchmark_df["Total Distance (km)"].notna()]
    if not valid.empty:
        best_row = valid.loc[valid["Total Distance (km)"].idxmin()]

        st.markdown(f"""
        <div class="benchmark-winner">
            <div class="benchmark-winner-icon">🏆</div>
            <div>
                <div class="benchmark-winner-text">
                    Best Algorithm: {best_row["Algorithm"]}
                </div>
                <div class="benchmark-winner-detail">
                    {best_row['Total Distance (km)']:.2f} km — {best_row['Runtime (seconds)']:.4f}s — {best_row['Vehicles Used']} vehicles
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    tab_bm_table, tab_bm_chart, tab_bm_conv = st.tabs([
        "📊 Results Table", "📊 Comparison Charts", "📈 Convergence"
    ])

    with tab_bm_table:
        st.dataframe(benchmark_df, use_container_width=True, hide_index=True)

    with tab_bm_chart:
        # Distance comparison
        algo_colors = {
            "QPSO": "#00d4ff", "PSO": "#7c3aed", "GA": "#00ff88",
            "ACO": "#ffaa00", "SA": "#ff6b9d", "Tabu Search": "#38bdf8",
            "ALNS": "#06d6a0"
        }

        fig_dist = go.Figure()
        for _, row in benchmark_df.iterrows():
            algo = row["Algorithm"]
            fig_dist.add_trace(go.Bar(
                x=[algo],
                y=[row["Total Distance (km)"]],
                name=algo,
                marker=dict(
                    color=algo_colors.get(algo, "#00d4ff"),
                    opacity=0.85,
                    line=dict(color='rgba(255,255,255,0.1)', width=1),
                ),
                text=f'{row["Total Distance (km)"]:.2f}',
                textposition='outside',
                textfont=dict(color='#94a3b8', size=12),
                hovertemplate=f'<b>{algo}</b><br>'
                              f'Distance: {row["Total Distance (km)"]:.2f} km<br>'
                              f'Runtime: {row["Runtime (seconds)"]:.4f}s<extra></extra>'
            ))

        fig_dist.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text="<b>Total Distance Comparison</b>", font=dict(color="#e8ecf4", size=14), x=0.02),
            yaxis_title="Distance (km)",
            height=380,
            showlegend=False,
            bargap=0.35,
        )
        st.plotly_chart(fig_dist, use_container_width=True)

        # Runtime comparison
        fig_time = go.Figure()
        for _, row in benchmark_df.iterrows():
            algo = row["Algorithm"]
            fig_time.add_trace(go.Bar(
                x=[algo],
                y=[row["Runtime (seconds)"]],
                name=algo,
                marker=dict(
                    color=algo_colors.get(algo, "#00d4ff"),
                    opacity=0.85,
                    line=dict(color='rgba(255,255,255,0.1)', width=1),
                ),
                text=f'{row["Runtime (seconds)"]:.4f}s',
                textposition='outside',
                textfont=dict(color='#94a3b8', size=12),
            ))

        fig_time.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text="<b>Runtime Comparison</b>", font=dict(color="#e8ecf4", size=14), x=0.02),
            yaxis_title="Runtime (seconds)",
            height=380,
            showlegend=False,
            bargap=0.35,
        )
        st.plotly_chart(fig_time, use_container_width=True)

    with tab_bm_conv:
        bm_hist = st.session_state.get("benchmark_histories", {})
        if bm_hist:
            fig_conv = go.Figure()
            for algo_name, hist in bm_hist.items():
                color = algo_colors.get(algo_name, "#00d4ff")
                fig_conv.add_trace(go.Scatter(
                    x=list(range(len(hist))),
                    y=hist,
                    mode='lines',
                    name=algo_name,
                    line=dict(color=color, width=2.5, shape='spline', smoothing=0.8),
                    hovertemplate=f'<b>{algo_name}</b><br>'
                                  'Iteration: %{x}<br>'
                                  'Distance: %{y:.3f} km<extra></extra>'
                ))

            fig_conv.update_layout(
                **PLOTLY_LAYOUT,
                title=dict(
                    text="<b>Convergence Comparison</b>",
                    font=dict(color="#e8ecf4", size=14),
                    x=0.02
                ),
                xaxis_title="Iteration",
                yaxis_title="Best Distance (km)",
                height=420,
                showlegend=True,
            )
            st.plotly_chart(fig_conv, use_container_width=True)
        else:
            st.info("No convergence data available.")


# ============================================================
# ALGORITHM INFORMATION
# ============================================================

st.markdown("""
<div class="section-header">
    <div class="section-icon">🧠</div>
    <div class="section-title">Algorithm Overview</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-subtitle">Understanding the quantum-inspired and metaheuristic optimization approaches</div>',
            unsafe_allow_html=True)

with st.expander("Learn about the optimization algorithms", expanded=False):
    a1, a2, a3 = st.columns(3)

    with a1:
        st.markdown("""
        <div class="algo-card algo-card-qpso">
            <div class="algo-card-icon">⚛️</div>
            <div class="algo-card-title algo-card-title-qpso">QPSO — Quantum-Inspired PSO</div>
            <div class="algo-card-desc">
                Combines <b>quantum mechanics principles</b> with particle swarm optimization.
                Uses a quantum probability distribution instead of deterministic
                velocity updates. The contraction-expansion coefficient (β) controls
                exploration vs exploitation.
            </div>
            <div class="algo-card-strength">
                <b>Strength:</b> <span style="color: #00d4ff;">Superior global search — avoids local optima</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with a2:
        st.markdown("""
        <div class="algo-card algo-card-pso">
            <div class="algo-card-icon">🌀</div>
            <div class="algo-card-title algo-card-title-pso">PSO — Particle Swarm Optimization</div>
            <div class="algo-card-desc">
                Swarm intelligence inspired by <b>bird flocking behavior</b>.
                Each particle has a position and velocity, updated based on
                personal best and global best. Uses inertia, cognitive,
                and social coefficients.
            </div>
            <div class="algo-card-strength">
                <b>Strength:</b> <span style="color: #7c3aed;">Fast convergence, simple to tune</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with a3:
        st.markdown("""
        <div class="algo-card algo-card-ga">
            <div class="algo-card-icon">🧬</div>
            <div class="algo-card-title algo-card-title-ga">GA — Genetic Algorithm</div>
            <div class="algo-card-desc">
                Evolutionary algorithm inspired by <b>natural selection</b>.
                Uses tournament selection, order crossover, and
                swap mutation. Elitism preserves the best solution
                across generations.
            </div>
            <div class="algo-card-strength">
                <b>Strength:</b> <span style="color: #00ff88;">Robust for combinatorial optimization</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# ORDERS & LOCATIONS TABLE
# ============================================================

st.markdown("""
<div class="section-header">
    <div class="section-icon">📍</div>
    <div class="section-title">Orders & Network Locations</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-subtitle">All delivery points with road network snap status</div>',
            unsafe_allow_html=True)

display_df = customers_df.copy()
display_df["road_node"] = [
    snapped_locations[row["id"]]["node"]
    if row["id"] in snapped_locations else None
    for _, row in customers_df.iterrows()
]
display_df["status"] = [
    "✅ Ready" if row["id"] in snapped_locations else "❌ Not Snapped"
    for _, row in customers_df.iterrows()
]

st.dataframe(display_df, use_container_width=True, hide_index=True)

# ---- Network Information ----
with st.expander("🔎 Network Information"):
    nc1, nc2 = st.columns(2)
    with nc1:
        st.write("**Map Location**")
        st.code(place, language=None)
        st.write("**Road Nodes**")
        st.write(f"{info['nodes']:,}")
        st.write("**Road Edges**")
        st.write(f"{info['edges']:,}")
    with nc2:
        st.write("**Graph Type**")
        st.write("Directed" if G.is_directed() else "Undirected")
        st.write("**Customers**")
        st.write(customer_count)
        st.write("**Locations Snapped**")
        st.write(f"{len(snapped_locations)} / {len(locations)}")


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown("""
<div class="footer-container">
    <div class="footer-brand">FLUXROUTE — Quantum Route Intelligence</div>
    <div class="footer-text">
        Quantum-Inspired Intelligent Traffic Route Optimization
        in Transportation Systems Using Metaheuristic Optimization
    </div>
    <div class="footer-links">
        <div class="footer-link">SIH26137</div>
        <div class="footer-link">Egreen Quanta</div>
        <div class="footer-link">Smart India Hackathon 2026</div>
    </div>
    <div class="footer-text" style="margin-top: 10px; font-size: 0.68rem;">
        Powered by OpenStreetMap &nbsp;•&nbsp;
        QPSO + PSO + Genetic Algorithm
        &nbsp;•&nbsp; Real-time Road Network Analysis
        &nbsp;•&nbsp; Python + Streamlit
    </div>
</div>
""", unsafe_allow_html=True)