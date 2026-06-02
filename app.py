import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import time

from agent import DischargeAgent
from learning import LearningLoop, SimulatedClinician

# Page Config
st.set_page_config(
    page_title="Dscribe Clinical Portal - Agentic Discharge Summaries",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling via CSS
st.markdown("""
<style>
    :root {
        --bg-primary: #0f172a;
        --bg-card: rgba(255,255,255,0.08);
        --text-primary: #f8fafc;
        --accent-primary: #0ea5e9; /* cyan */
        --accent-secondary: #f59e0b; /* amber */
        --font-family: 'Inter', sans-serif;
    }
    body, .main, .stApp {
        background: var(--bg-primary);
        color: var(--text-primary);
        font-family: var(--font-family);
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e293b;
        color: var(--text-primary);
    }
    .stTabs [aria-selected="true"] {
        background-color: var(--accent-primary) !important;
        color: #ffffff !important;
    }
    h1, h2, h3 {
        color: #ffffff !important;
        font-family: var(--font-family);
    }
    .card {
        background: var(--bg-card);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        color: var(--text-primary) !important;
        backdrop-filter: blur(12px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.4);
    }
    button, button p, button div, button span {
        background-color: var(--accent-primary) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        border-radius: 6px;
        transition: background-color 0.2s ease, transform 0.2s ease;
    }
    button:hover {
        background-color: var(--accent-secondary) !important;
        transform: translateY(-2px);
    }
    /* Keep other existing rules (inputs, code, sidebar) unchanged but ensure they use vars */
    .stTextInput input, .stTextArea textarea {
        background-color: #1e293b !important;
        color: var(--text-primary) !important;
    }
    code {
        color: var(--text-primary) !important;
        background-color: #0f172a !important;
    }
    [data-testid="stSidebar"] {
        background-color: #1e293b !important;
    }
    .stLabel {
        color: var(--text-primary) !important;
    }
</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
/* High‑contrast for all text elements */
.stApp, .main, .stMarkdown, .stCaption, .stText, .stInfo, .stError, .stSuccess, .stWarning, .stTable, .stDataFrame, .stHeader, .stTabs, .stSelectbox label, .stNumberInput label, .stRadio label {
    color: #f8fafc !important;
}
/* Force all markdown descendants to white */
.stMarkdown * {
    color: #f8fafc !important;
    word-wrap: break-word !important;
    overflow-wrap: anywhere !important;
}
/* Links */
.stMarkdown a, .stLink {
    color: #38bdf8 !important;
    text-decoration: underline;
}
/* Ensure boxes have white text */
.stInfo, .stSuccess, .stWarning, .stError {
    color: #f8fafc !important;
}
</style>
""", unsafe_allow_html=True)

# App Title & Header
st.title("🩺 Dscribe Clinical Portal")
st.subheader("Agentic AI-Driven Safe Discharge Summaries with Doctor-in-the-Loop Learning")

# Sidebar settings
st.sidebar.image("https://img.icons8.com/color/96/medical-doctor.png", width=80)
st.sidebar.header("Agent Parameters")
patient_selection = st.sidebar.selectbox("Active Patient Record", ["Patient 2 (Acute Appendicitis)"])
max_steps = st.sidebar.slider("Agent Execution Step Cap", 5, 20, 10)

st.sidebar.markdown("---")
st.sidebar.subheader("Doctor Learning Loop (Part 2)")
reset_memory = st.sidebar.button("Clear Learned Corrections")
if reset_memory:
    LearningLoop.clear_memory()
    st.sidebar.success("Learned memory cleared! Restarting baseline iteration 1.")
    st.rerun()

# Load correction history to display accuracy curves
history_file = "accuracy_history.json"
def load_history():
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            return json.load(f)
    return []

def save_history(history):
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=4)

history = load_history()
iteration_num = len(history) + 1

# Initialize Agent
agent = DischargeAgent(pdf_path="patient 2 (1).pdf", max_steps=max_steps)

# Run Ingestion
with st.spinner("Analyzing and parsing patient PDF..."):
    ingest_success = agent.run_ingestion()

if not ingest_success:
    st.error("Failed to ingest PDF. Please check your API key in the .env file!")
    st.stop()

# Run the agent reasoning loop and collect outputs
@st.cache_data(ttl=600)
def run_agent_loop(iter_token):
    # Pass a token to force cached rerun on iteration increment
    return agent.execute_loop()

results = run_agent_loop(iteration_num)
draft = results["draft"]
trace = results["trace"]
recons = results["reconciliations"]
interactions = results["interactions"]
escalations = results["escalations"]

# Setup main tabs
tab_review, tab_reconcile, tab_trace, tab_learning = st.tabs([
    "🔍 Clinician Review & Draft", 
    "💊 Medication Reconciliation & Safety", 
    "🧠 Step-by-Step Agent Trace", 
    "📈 Doctor Edit Feedback Loop"
])

# --- TAB 1: CLINICIAN REVIEW & DRAFT ---
with tab_review:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Generated Discharge Summary Draft")
        st.markdown("*Review the draft below. If anything needs correction, feel free to edit it directly.*")
        
        # Clinician editable form fields
        with st.form("draft_edit_form"):
            demo_val = st.text_area("Patient Demographics", draft.get("demographics", ""), height=120)
            diagnosis_val = st.text_area("Principal and Secondary Diagnoses", draft.get("diagnoses", ""), height=100)
            course_val = st.text_area("Hospital Course Summary", draft.get("hospital_course", ""), height=150)
            procedure_val = st.text_area("Procedures Performed", draft.get("procedures", ""), height=100)
            med_val = st.text_area("Discharge Medications & Instructions", draft.get("medications", ""), height=150)
            allergy_val = st.text_area("Allergies", draft.get("allergies", ""), height=80)
            pending_val = st.text_area("Pending Laboratory & Imaging Results", draft.get("pending_results", ""), height=80)
            condition_val = st.text_area("Condition at Discharge", draft.get("discharge_condition", ""), height=80)
            
            submit_edits = st.form_submit_button("Submit Corrections & Approve Draft")
            
            if submit_edits:
                # Save corrections into memory (Part 2 Learning)
                new_draft = {
                    "demographics": demo_val,
                    "diagnoses": diagnosis_val,
                    "hospital_course": course_val,
                    "procedures": procedure_val,
                    "medications": med_val,
                    "allergies": allergy_val,
                    "pending_results": pending_val,
                    "discharge_condition": condition_val
                }
                
                # Compare original draft against doctor edits to record learning signals
                total_draft_text = " ".join(draft.values())
                total_edited_text = " ".join(new_draft.values())
                
                accuracy = LearningLoop.get_accuracy_signal(total_draft_text, total_edited_text)
                
                # Save each edited section in structured correction memory
                for sec_key in draft.keys():
                    if draft[sec_key] != new_draft[sec_key]:
                        LearningLoop.save_correction(sec_key, draft[sec_key], new_draft[sec_key])
                        
                # Update history log
                history.append({
                    "iteration": iteration_num,
                    "accuracy": accuracy,
                    "edit_distance": LearningLoop.calculate_levenshtein(total_draft_text, total_edited_text)
                })
                save_history(history)
                
                st.success(f"Draft approved! Edit similarity of {accuracy:.1%} recorded. Agent is loading correction memory to optimize next run!")
                time.sleep(1.5)
                st.rerun()
                
    with col2:
        st.subheader("⚠️ Critical Safety & Escalations")
        st.markdown("*Automatic alerts generated by the agent loop due to conflicts or high-risk findings.*")
        
        if escalations:
            for esc in escalations:
                if "ALLERGY" in esc:
                    st.error(esc)
                elif "WARNING" in esc:
                    st.warning(esc)
                else:
                    st.info(esc)
        else:
            st.success("No critical clinical concerns flagged. Draft is marked fully safe.")
            
        # Clinical Guardrails Info
        st.markdown("""
        <div class="card">
            <h4>No-Fabrication Guardrails Active</h4>
            <p style="font-size: 14px; color: #94a3b8;">
                The agent enforces a strict grounding checker. If patient demographics (Name, DOB) are not documented in the raw source files, they are explicitly marked <b>MISSING</b> to prevent hallucination.
            </p>
        </div>
        """, unsafe_allow_html=True)


# --- TAB 2: MEDICATION RECONCILIATION & SAFETY ---
with tab_reconcile:
    st.subheader("Medication Reconciliation")
    st.markdown("Comparing the patient's **Admission Medications** against their **Discharge Medications**.")
    
    # Format Reconciliation Table
    recon_df = pd.DataFrame(recons)
    if not recon_df.empty:
        # Style type column
        def color_type(val):
            if val == "Added":
                return 'background-color: #064e3b; color: #34d399; font-weight: bold;'
            elif val == "Stopped":
                return 'background-color: #7f1d1d; color: #f87171; font-weight: bold;'
            elif val == "Changed":
                return 'background-color: #78350f; color: #fbbf24; font-weight: bold;'
            return 'color: #94a3b8;'
            
        styled_df = recon_df.style.map(color_type, subset=['type'])
        st.dataframe(styled_df, width="stretch")
    else:
        st.info("No medication changes detected.")
        
    st.markdown("---")
    st.subheader("🔬 Real-time NIH RxNav Drug-Drug Interaction Report")
    
    if interactions:
        for idx, inter in enumerate(interactions):
            st.error(f"⚠️ **Severe Interaction {idx+1}:** {inter['description']} (Involving: {', '.join(inter['drugs'])})")
            st.caption(f"Source: {inter['source']}")
    else:
        st.success("No severe drug-drug interactions detected between current discharge medications.")


# --- TAB 3: STEP-BY-STEP AGENT TRACE ---
with tab_trace:
    st.subheader("Observable Agent Execution Trace")
    st.markdown("This timeline logs the agent's real-time **Reasoning $\\rightarrow$ Action chosen $\\rightarrow$ Inputs $\\rightarrow$ Result** workflow.")
    
    for step in trace:
        st.markdown(f"""
        <div class="card" style="border-left: 5px solid #0d9488;">
            <span style="background-color: #0d9488; color: white; padding: 4px 10px; border-radius: 4px; font-weight: bold; font-size: 12px;">STEP {step['step']}</span>
            <p style="margin-top: 10px;"><b>Reasoning Thought:</b> {step['reasoning']}</p>
            <p><b>Action / Tool Chosen:</b> <code style="color: #38bdf8;">{step['action']}</code> (Inputs: <code>{step['inputs']}</code>)</p>
            <p style="background-color: #0f172a; padding: 12px; border-radius: 6px; border: 1px solid #1e293b; color: #a5f3fc;"><b>Tool Output Result:</b> {step['result']}</p>
            <p style="color: #2dd4bf;"><b>Next Decision:</b> {step['next_decision']}</p>
        </div>
        """, unsafe_allow_html=True)


# --- TAB 4: DOCTOR EDIT FEEDBACK LOOP ---
with tab_learning:
    st.subheader("Part 2: Active Learning from Doctor Corrections")
    st.markdown("This loop evaluates how the agent adapts to your specific style and corrections over multiple iterations.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown(f"**Current Run Iteration:** {iteration_num}")
        
        # Display Accuracy Stats
        if history:
            current_acc = history[-1]["accuracy"]
            st.markdown(f"#### Latest Draft Edit Similarity: <span class='metric-val'>{current_acc:.1%}</span>", unsafe_allow_html=True)
            st.caption("A higher score means less manual editing was required by the clinician (lower edit burden).")
        else:
            st.markdown("#### Latest Draft Edit Similarity: <span class='metric-val'>Baseline (TBD)</span>", unsafe_allow_html=True)
            st.caption("Approve the current draft with changes in Tab 1 to record the baseline similarity!")
            
        # Display active corrections in memory
        memory = LearningLoop.load_memory()
        st.markdown("#### Learned Styles & Active Preferences")
        if memory:
            for sec, rule in memory.items():
                st.info(f"**Rule for {sec.capitalize()}:** In future summaries, the agent will adapt its style to match your corrections for this section.")
        else:
            st.markdown("*No corrections saved yet. Complete a review cycle to train the agent.*")
            
    with col2:
        st.markdown("#### Clinician Edit Similarity Curve")
        
        if len(history) > 0:
            # Build and display plot
            df_hist = pd.DataFrame(history)
            
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor('#1e293b')
            ax.set_facecolor('#0f172a')
            
            ax.plot(df_hist["iteration"], df_hist["accuracy"] * 100, marker='o', linewidth=3, color='#2dd4bf', label="Edit Similarity")
            ax.set_title("Clinician Alignment Curve (Higher = Fewer Edits Needed)", color='#f8fafc', fontsize=12)
            ax.set_xlabel("Iteration / Review Cycle", color='#94a3b8')
            ax.set_ylabel("Similarity (%)", color='#94a3b8')
            ax.tick_params(colors='#94a3b8')
            ax.set_ylim(0, 105)
            
            # Show grid lines
            ax.grid(True, color='#334155', linestyle='--', alpha=0.5)
            
            st.pyplot(fig)
        else:
            # Demo chart to show user what to expect
            st.info("Curve will appear here once you approve a draft with edits!")
            
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor('#1e293b')
            ax.set_facecolor('#0f172a')
            
            x = [1, 2, 3]
            y = [40, 85, 98]
            ax.plot(x, y, marker='o', color='#475569', linestyle='--')
            ax.set_title("Example Alignment Curve (Ideal Progression)", color='#475569')
            ax.set_ylim(0, 105)
            ax.grid(True, color='#1e293b')
            st.pyplot(fig)
