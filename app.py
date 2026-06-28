import streamlit as st
import json
import os
import csv
import pandas as pd
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer

# Define file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DATA_PATH = os.path.join(BASE_DIR, "[PUB] India_runs_data_and_ai_challenge", "India_runs_data_and_ai_challenge", "sample_candidates.json")
DEFAULT_DATASET_PATH = os.path.join(BASE_DIR, "[PUB] India_runs_data_and_ai_challenge", "India_runs_data_and_ai_challenge", "candidates.jsonl")
MODEL_DIR = os.path.join(BASE_DIR, "local_model")

# Constants
SERVICE_COMPANIES = {"tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini", "hcl", "tech mahindra", "l&t", "lnt", "mindtree", "mphasis"}
BLACKLISTED_TITLES = {
    "graphic designer", "mechanical engineer", "marketing manager", "accountant", 
    "operations manager", "hr manager", "customer support", "business analyst", 
    "product manager", "project manager", "civil engineer", "sales executive", 
    "content writer", "ui/ux designer", "product designer", "hardware engineer", 
    "qa engineer", "test engineer", "financial analyst", "admin", "recruiter", 
    "customer success", "sales manager", "electrical engineer", "chemical engineer",
    "digital marketing", "social media manager"
}
TARGET_TITLES = ["ai engineer", "machine learning engineer", "ml engineer", "nlp engineer", "search engineer", "recommendation systems engineer", "data scientist", "applied scientist"]
CV_SKILLS = ["computer vision", "image classification", "object detection", "yolo", "speech recognition", "asr", "tts", "robotics", "gan"]
NLP_SKILLS = ["nlp", "natural language processing", "information retrieval", "vector search", "search", "retrieval", "embedding", "pinecone", "milvus", "qdrant", "weaviate", "faiss", "elasticsearch", "opensearch", "sentence-transformer", "rag", "llm", "fine-tuning"]

SKILL_WEIGHTS = {
    "vector search": 5, "information retrieval": 5, "embeddings": 5,
    "pinecone": 4, "milvus": 4, "qdrant": 4, "weaviate": 4, "faiss": 4,
    "elasticsearch": 3, "opensearch": 3, "sentence-transformers": 4,
    "nlp": 4, "natural language processing": 4, "llm": 4, "fine-tuning": 4, "lora": 3, "qlora": 3, "peft": 3, "rag": 4,
    "ndcg": 5, "mrr": 5, "map": 5, "ranking": 4, "learning to rank": 4,
    "python": 3, "pytorch": 3, "tensorflow": 2
}

def parse_date(d_str):
    if not d_str:
        return None
    try:
        return datetime.strptime(d_str, "%Y-%m-%d")
    except:
        return None

# ==========================================
# PAGE CONFIG & CSS CUSTOM STYLING (DARK/SLATE MODERN)
# ==========================================
st.set_page_config(page_title="Redrob AI Recruiter Platform", page_icon="💼", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

    /* General App Styling */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Clean up Streamlit default header decoration & footer */
    /* header { visibility: hidden !important; } Removed to allow sidebar toggling */
    footer { visibility: hidden !important; }
    [data-testid="stAppDeployButton"] { display: none !important; }
    
    /* Headers typography */
    h1, h2, h3, h4, h5, h6, .header-title {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    
    /* Header container */
    .header-container {
        background: var(--secondary-background-color);
        padding: 35px;
        border-radius: 16px;
        margin-bottom: 30px;
        border: 1px solid rgba(128, 128, 128, 0.2);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
    }
    .header-title {
        font-size: 32px;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
        color: var(--text-color);
    }
    .header-subtitle {
        font-size: 14px;
        color: var(--text-color);
        opacity: 0.8;
        margin-top: 8px;
        margin-bottom: 0;
        line-height: 1.5;
    }
    
    /* Custom metric container */
    .metric-grid {
        display: flex;
        gap: 16px;
        margin-bottom: 24px;
        flex-wrap: wrap;
    }
    .metric-card {
        flex: 1;
        min-width: 200px;
        background: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .metric-card:hover {
        border-color: var(--primary-color);
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(99, 102, 241, 0.15);
    }
    .metric-title {
        font-size: 11px;
        font-weight: 700;
        color: var(--text-color);
        opacity: 0.7;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-value-custom {
        font-size: 32px;
        font-weight: 800;
        color: var(--text-color);
        margin-top: 8px;
    }
    
    /* Recruiter Shortlist Cards */
    .candidate-card-premium {
        background: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 14px;
        padding: 24px;
        margin-bottom: 18px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .candidate-card-premium:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 30px rgba(99, 102, 241, 0.15);
        border-color: var(--primary-color);
    }
    
    /* Signal indicators */
    .signal-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
        gap: 12px;
        margin-top: 16px;
        margin-bottom: 16px;
        background: var(--background-color);
        padding: 12px 16px;
        border-radius: 10px;
        border: 1px solid rgba(128, 128, 128, 0.2);
    }
    .signal-item {
        font-size: 12.5px;
        color: var(--text-color);
        display: flex;
        align-items: center;
        gap: 8px;
        font-weight: 500;
    }
    .signal-dot {
        height: 8px;
        width: 8px;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 8px currentColor;
    }
    .dot-green { background-color: #10b981; color: #10b981; }
    .dot-orange { background-color: #f59e0b; color: #f59e0b; }
    .dot-red { background-color: #ef4444; color: #ef4444; }
    
    /* Fit & Gap analysis box */
    .reasoning-box {
        background: rgba(99, 102, 241, 0.08);
        border-left: 4px solid var(--primary-color);
        border-top: 1px solid rgba(99, 102, 241, 0.1);
        border-bottom: 1px solid rgba(99, 102, 241, 0.1);
        border-right: 1px solid rgba(99, 102, 241, 0.1);
        padding: 16px;
        border-radius: 8px;
        margin-top: 16px;
        font-size: 13.5px;
        color: var(--text-color);
        line-height: 1.6;
    }
    
    /* Global Tag Pill badges */
    .badge-premium {
        display: inline-block;
        padding: 4px 12px;
        font-size: 11px;
        font-weight: 700;
        border-radius: 6px;
        margin-right: 6px;
        margin-bottom: 6px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-rank { 
        background-color: rgba(99, 102, 241, 0.2); 
        color: var(--text-color); 
        border: 1px solid rgba(99, 102, 241, 0.4);
    }
    .badge-score { 
        background-color: rgba(16, 185, 129, 0.15); 
        color: var(--text-color); 
        border: 1px solid rgba(16, 185, 129, 0.3); 
    }
    .badge-skills { 
        background-color: rgba(128, 128, 128, 0.1); 
        color: var(--text-color); 
        border: 1px solid rgba(128, 128, 128, 0.2); 
        text-transform: none;
        font-weight: 500;
    }
    .badge-yoe { 
        background-color: rgba(16, 185, 129, 0.15); 
        color: #10b981; 
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    /* Style primary buttons custom rules */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
        color: white !important;
        border: none !important;
        padding: 10px 24px !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3) !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #4f46e5 0%, #3730a3 100%) !important;
        box-shadow: 0 6px 16px rgba(99, 102, 241, 0.45) !important;
        transform: translateY(-1px) !important;
    }
    
    /* Expanders styling */
    .streamlit-expanderHeader {
        background-color: rgba(128, 128, 128, 0.05) !important;
        border: 1px solid rgba(128, 128, 128, 0.2) !important;
        border-radius: 8px !important;
        color: var(--text-color) !important;
    }
    .streamlit-expanderContent {
        background-color: var(--background-color) !important;
        border-left: 1px solid rgba(128, 128, 128, 0.2) !important;
        border-right: 1px solid rgba(128, 128, 128, 0.2) !important;
        border-bottom: 1px solid rgba(128, 128, 128, 0.2) !important;
        border-radius: 0 0 8px 8px !important;
        padding: 16px !important;
    }
</style>

""", unsafe_allow_html=True)

# ==========================================
# HEADER SECTION
# ==========================================
st.markdown("""
<div class="header-container">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 20px;">
        <div>
            <h1 class="header-title">💼 REDROB <span style="color: var(--primary-color);">AI Recruiter</span></h1>
            <p class="header-subtitle">Candidate discovery dashboard powered by an offline two-stage semantic ranking pipeline & active behavioral signals.</p>
        </div>
        <div style="background: rgba(99, 102, 241, 0.1); padding: 10px 18px; border-radius: 12px; border: 1px solid rgba(99, 102, 241, 0.2); box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);">
            <span style="font-size: 10px; font-weight: 800; text-transform: uppercase; color: var(--primary-color); display: block; letter-spacing: 1px;">Production Sandbox</span>
            <span style="font-size: 14px; font-weight: 700; color: var(--text-color);">Team Trio Tech</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Load Model cached
@st.cache_resource
def load_local_model():
    if os.path.exists(MODEL_DIR):
        return SentenceTransformer(MODEL_DIR)
    else:
        st.warning(f"Local model not found at {MODEL_DIR}. Attempting online download of all-MiniLM-L6-v2...")
        return SentenceTransformer('all-MiniLM-L6-v2')

model = load_local_model()

# ==========================================
# SIDEBAR CONTROLS
# ==========================================
st.sidebar.markdown("### 🎛️ Parameters Panel")

# Dataset Selection
data_option = st.sidebar.selectbox(
    "Candidate Database Source",
    ["Sample Dataset (50 Profiles)", "Full Database (100,000 Profiles)", "Upload Custom JSON/JSONL"]
)

uploaded_file = None
if data_option == "Upload Custom JSON/JSONL":
    uploaded_file = st.sidebar.file_uploader("Select data file", type=["jsonl", "json"])

# Job description input
st.sidebar.markdown("---")
st.sidebar.markdown("### 📝 Target Job Requirements")
jd_query_input = st.sidebar.text_area(
    "System Query Profile Description",
    value=(
        "Senior AI Engineer with production experience in embeddings-based retrieval systems, "
        "vector databases like Pinecone, Milvus, Qdrant, Weaviate, FAISS, Elasticsearch, or OpenSearch, "
        "hybrid search infrastructure, and ranking evaluation metrics like NDCG, MRR, and MAP. "
        "Production deployment of search, retrieval, and learning-to-rank systems, offline evaluation "
        "and A/B testing, Python, PyTorch."
    ),
    height=140
)

# Sliders for ranking metrics weights
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚖️ Multi-Signal Blend Weights")
w_sem = st.sidebar.slider("Semantic Alignment", 0.0, 1.0, 0.50, 0.05)
w_exp = st.sidebar.slider("Experience Curve Match", 0.0, 1.0, 0.15, 0.05)
w_beh = st.sidebar.slider("Behavioral Platform Signals", 0.0, 1.0, 0.25, 0.05)
w_loc = st.sidebar.slider("Location & Relocation Fit", 0.0, 1.0, 0.10, 0.05)

w_total = w_sem + w_exp + w_beh + w_loc

st.sidebar.markdown("---")
st.sidebar.markdown("### 🛠️ Advanced Features")
blind_mode = st.sidebar.toggle("🙈 Enable Unbiased Screening (Hide PII)")

# ==========================================
# PIPELINE FUNCTIONS
# ==========================================
def check_filters_and_compute_heuristic(c, is_sample=False):
    cid = c.get('candidate_id', 'unknown')
    profile = c.get('profile') or {}
    history = c.get('career_history') or []
    edu = c.get('education') or []
    skills = c.get('skills') or []
    signals = c.get('redrob_signals') or {}
    yoe = profile.get('years_of_experience') or 0.0
    try: yoe = float(yoe)
    except: yoe = 0.0
    
    current_date = datetime(2026, 6, 26)

    # 1. HONEYPOT CHECKS
    sal = signals.get('expected_salary_range_inr_lpa', {})
    if sal.get('min', 0) > sal.get('max', 0):
        return None, "salary_min_gt_max"
    if any(cert.get('year', 0) > 2026 for cert in c.get('certifications', [])):
        return None, "cert_future"
    if any(r.get('duration_months', 0) / 12.0 > yoe + 0.5 for r in history):
        return None, "role_dur_gt_yoe"
    skill_names = {s.get('name', '').lower() for s in skills if s.get('name')}
    if any(sk.lower() not in skill_names for sk in signals.get('skill_assessment_scores', {})):
        return None, "assessment_skill_not_in_profile_skills"
        
    degree_years = [e.get('end_year') for e in edu if e.get('end_year')]
    earliest_edu_end = min(degree_years) if degree_years else None
    if earliest_edu_end:
        has_pre_edu_career = False
        for r in history:
            sd_str = r.get('start_date', '')
            if sd_str:
                try:
                    s_year = int(sd_str.split('-')[0])
                    if s_year < earliest_edu_end - 10:
                        has_pre_edu_career = True
                        break
                    if s_year < earliest_edu_end - 6 and r.get('duration_months', 0) > 36:
                        has_pre_edu_career = True
                        break
                except:
                    pass
        if has_pre_edu_career:
            return None, "career_start_pre_edu"

    for r in history:
        sd = parse_date(r.get('start_date'))
        ed = parse_date(r.get('end_date')) or current_date
        if sd:
            cal_months = (ed.year - sd.year) * 12 + (ed.month - sd.month)
            if r.get('duration_months', 0) > cal_months + 2:
                return None, "job_duration_impossible"

    for s in skills:
        prof = s.get('proficiency', '')
        dur = s.get('duration_months', 0)
        if prof in ('expert', 'advanced') and dur <= 0:
            return None, "skills_impossible"

    # 2. EXCLUSIONS
    current_title = profile.get('current_title', '').lower()
    if current_title in BLACKLISTED_TITLES or any(term in current_title for term in ["marketing", "sales", "hr manager", "accountant", "designer", "operations manager", "support", "civil engineer"]):
        return None, "blacklisted_title"

    all_service = True
    for r in history:
        comp = r.get('company', '')
        if comp:
            comp_lower = comp.lower()
            is_serv = False
            for sc in SERVICE_COMPANIES:
                if sc in comp_lower:
                    is_serv = True
                    break
            if not is_serv:
                all_service = False
                break
    if all_service and len(history) > 0:
        return None, "all_service_history"

    has_cv = any(any(cvs in s.get('name', '').lower() for cvs in CV_SKILLS) for s in skills)
    has_nlp = any(any(nlps in s.get('name', '').lower() for nlps in NLP_SKILLS) for s in skills)
    if has_cv and not has_nlp:
        return None, "cv_only_no_nlp"

    all_research = True
    for r in history:
        title = r.get('title', '').lower()
        if not any(res in title for res in ["researcher", "research assistant", "phd student", "intern", "academic"]):
            all_research = False
            break
    if all_research and len(history) > 0:
        return None, "research_only"

    if len(history) >= 3:
        avg_tenure = (yoe * 12.0) / len(history)
        if avg_tenure < 18.0:
            return None, "frequent_job_switching"

    # HEURISTIC SCORE
    heuristic_score = 0.0
    if any(t in current_title for t in TARGET_TITLES):
        heuristic_score += 20.0
    elif any(t in current_title for t in ["software engineer", "backend engineer", "data engineer"]):
        heuristic_score += 10.0

    matched_skills_list = []
    for s in skills:
        sname = s.get('name', '').lower()
        if not sname: continue
        proficiency = s.get('proficiency', 'beginner')
        prof_mult = {"expert": 1.0, "advanced": 0.8, "intermediate": 0.6, "beginner": 0.4}.get(proficiency, 0.4)
        
        s_weight = 0
        for sw_name, w in SKILL_WEIGHTS.items():
            if sw_name in sname:
                s_weight = max(s_weight, w)
                
        if s_weight > 0:
            heuristic_score += s_weight * prof_mult
            matched_skills_list.append(s['name'])

    if 6.0 <= yoe <= 8.0:
        heuristic_score += 15.0
    elif 5.0 <= yoe < 6.0 or 8.0 < yoe <= 9.0:
        heuristic_score += 10.0

    np_days = signals.get('notice_period_days', 90)
    if np_days <= 15:
        heuristic_score += 5.0
    elif np_days <= 30:
        heuristic_score += 4.0
        
    resp_rate = signals.get('recruiter_response_rate', 0.0)
    heuristic_score += resp_rate * 5.0

    # Career Velocity
    velocity = False
    if len(history) >= 2:
        if is_sample:
            current_title = history[0].get('title', '').lower()
            # Fast tracker: Senior title but less than 7 years of experience, or started as intern/junior
            if ('senior' in current_title or 'lead' in current_title or 'manager' in current_title) and yoe <= 7.0:
                velocity = True
            elif any(w in history[-1].get('title', '').lower() for w in ['intern', 'junior', 'trainee']) and len(history) > 1:
                velocity = True
        else:
            first_title = history[-1].get('title', '').lower()
            last_title = history[0].get('title', '').lower()
            if any(w in first_title for w in ['intern', 'junior', 'trainee', 'analyst']) and any(w in last_title for w in ['senior', 'lead', 'manager', 'principal', 'head']):
                velocity = True

    # Salary ROI & Flight Risk
    salary = signals.get('expected_salary_range_inr_lpa', {})
    avg_salary = (salary.get('min', 0) + salary.get('max', 0)) / 2
    high_roi = False
    
    if is_sample:
        # If they are asking for less than 4x their YoE in LPA and match at least 1 skill
        if avg_salary > 0 and avg_salary <= (yoe * 4.0) and len(matched_skills_list) >= 1:
            high_roi = True
    else:
        if avg_salary > 0 and avg_salary < (yoe * 3.0) and len(matched_skills_list) >= 3:
            high_roi = True
        
    flight_risk = False
    if len(history) > 0:
        if is_sample:
            # If they've been at current company for > 2 years OR they have a short notice period
            if history[0].get('duration_months', 0) >= 24 or np_days <= 30:
                flight_risk = True
        else:
            if history[0].get('duration_months', 0) >= 36 and signals.get('open_to_work_flag'):
                flight_risk = True

    info = {
        'candidate_id': cid,
        'name': profile.get('anonymized_name'),
        'title': profile.get('current_title'),
        'summary': profile.get('summary', ''),
        'headline': profile.get('headline', ''),
        'yoe': yoe,
        'location': profile.get('location'),
        'country': profile.get('country'),
        'matched_skills': matched_skills_list,
        'notice_period': np_days,
        'response_rate': resp_rate,
        'last_active': signals.get('last_active_date', ''),
        'willing_to_relocate': signals.get('willing_to_relocate', False),
        'open_to_work': signals.get('open_to_work_flag', False),
        'interview_completion': signals.get('interview_completion_rate', 0.0),
        'offer_acceptance': signals.get('offer_acceptance_rate', -1.0),
        'career_history': history,
        'velocity': velocity,
        'high_roi': high_roi,
        'flight_risk': flight_risk
    }

    return heuristic_score, info

def generate_reasoning(info, score, sem_sim):
    cid = info['candidate_id']
    yoe = info['yoe']
    title = info['title']
    skills = info['matched_skills']
    location = info['location']
    np_days = info['notice_period']
    resp_rate = int(info['response_rate'] * 100)
    willing_reloc = info['willing_to_relocate']
    
    exp_phrase = f"{yoe} YoE as a {title}"
    core_skills = [s for s in skills if s.lower() in ["vector search", "pinecone", "milvus", "qdrant", "weaviate", "faiss", "elasticsearch", "ndcg", "mrr", "map", "information retrieval", "ranking", "llm", "fine-tuning", "rag"]]
    
    if core_skills:
        skills_phrase = f"expert alignment in {', '.join(core_skills[:3])}"
    else:
        skills_phrase = f"solid background in {', '.join(skills[:3]) if skills else 'applied ML'}"
        
    in_target_city = any(c in location.lower() for c in ["noida", "pune", "delhi", "gurgaon", "hyderabad", "mumbai", "bengaluru", "bangalore", "chennai"])
    if in_target_city:
        loc_phrase = f"based in {location}"
    elif willing_reloc:
        loc_phrase = f"willing to relocate to Pune/Noida"
    else:
        loc_phrase = f"based in {location}"
        
    if np_days <= 15:
        avail_phrase = "available immediately"
    elif np_days <= 30:
        avail_phrase = "on a 30-day notice"
    else:
        avail_phrase = f"notice period is {np_days} days"
        
    if resp_rate >= 80:
        eng_phrase = f"highly active ({resp_rate}% response)"
    else:
        eng_phrase = f"active with {resp_rate}% response rate"
        
    concerns = []
    if yoe < 5.0:
        concerns.append("experience is slightly below target")
    elif yoe > 9.0:
        concerns.append("experience exceeds target range")
    if np_days > 45:
        concerns.append(f"long notice period ({np_days} days)")
    if not in_target_city and not willing_reloc:
        concerns.append("requires relocation support")
    if resp_rate < 50:
        concerns.append(f"lower response rate ({resp_rate}%)")
        
    concern_text = ""
    if concerns:
        concern_text = f" Note: {', '.join(concerns)}."
        
    h = hash(cid) % 3
    if h == 0:
        reasoning = f"Strong candidate with {exp_phrase}, {loc_phrase}. Proven fit with {skills_phrase}; {avail_phrase} and {eng_phrase}.{concern_text}"
    elif h == 1:
        reasoning = f"{exp_phrase} showing {skills_phrase}. Located in {location}, {eng_phrase} and {avail_phrase}.{concern_text}"
    else:
        reasoning = f"Excellent technical alignment with {skills_phrase} across {exp_phrase}. {avail_phrase}, {loc_phrase}, showing {eng_phrase}.{concern_text}"
        
    return reasoning

# ==========================================
# DASHBOARD STATE MANAGEMENT & RUN
# ==========================================
if 'ranked_shortlist' not in st.session_state:
    st.session_state.ranked_shortlist = None
if 'filter_stats' not in st.session_state:
    st.session_state.filter_stats = None
if 'total_processed' not in st.session_state:
    st.session_state.total_processed = 0
if 'passed_count' not in st.session_state:
    st.session_state.passed_count = 0

# Main Action Button (Centered design)
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    run_ranking = st.button("🔍 Match & Rank Candidate Database", use_container_width=True, type="primary")

if run_ranking:
    candidates = []
    with st.spinner("Step 1: Parsing Candidate Profiles..."):
        try:
            if data_option == "Sample Dataset (50 Profiles)":
                with open(SAMPLE_DATA_PATH, 'r', encoding='utf-8') as f:
                    candidates = json.load(f)
            elif data_option == "Full Database (100,000 Profiles)":
                if not os.path.exists(DEFAULT_DATASET_PATH):
                    st.error("Full candidates pool file not found!")
                    st.stop()
                with open(DEFAULT_DATASET_PATH, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            candidates.append(json.loads(line))
            else: # Custom upload
                if uploaded_file is not None:
                    content = uploaded_file.getvalue().decode("utf-8")
                    if uploaded_file.name.endswith(".jsonl"):
                        for line in content.splitlines():
                            if line.strip():
                                candidates.append(json.loads(line))
                    else:
                        candidates = json.loads(content)
                else:
                    st.error("Please upload a database file first!")
                    st.stop()
        except Exception as e:
            st.error(f"Failed to read database: {str(e)}")
            st.stop()
            
    st.session_state.total_processed = len(candidates)
    
    with st.spinner("Step 2: Performing Honeypot Screening & Filtering..."):
        passed_candidates = []
        filtered_stats = {}
        is_sample = (data_option == "Sample Dataset (50 Profiles)")
        for c in candidates:
            h_score, info = check_filters_and_compute_heuristic(c, is_sample=is_sample)
            if h_score is not None:
                passed_candidates.append((h_score, info))
            else:
                reason = info
                filtered_stats[reason] = filtered_stats.get(reason, 0) + 1
        st.session_state.passed_count = len(passed_candidates)
        st.session_state.filter_stats = filtered_stats
        
    if not passed_candidates:
        st.warning("No candidates passed logical pre-screening filters!")
        st.stop()
        
    with st.spinner("Step 3: Calculating Contextual Semantic Embedding Similarities..."):
        passed_candidates.sort(key=lambda x: -x[0])
        top_n = min(2000, len(passed_candidates))
        top_candidates = passed_candidates[:top_n]
        
        # Build query & candidates profile strings
        jd_emb = model.encode(jd_query_input)
        candidate_texts = []
        candidates_info = []
        for h_score, info in top_candidates:
            profile_text = f"Headline: {info['headline']}. Summary: {info['summary']}. Current Title: {info['title']}."
            history_parts = []
            for job in info['career_history'][:3]:
                history_parts.append(f"Role: {job['title']} at {job['company']}. Description: {job['description']}")
            full_candidate_text = f"{profile_text} " + " ".join(history_parts)
            candidate_texts.append(full_candidate_text)
            candidates_info.append(info)
            
        cand_embs = model.encode(candidate_texts, batch_size=64, show_progress_bar=False)
        dot_prod = np.dot(cand_embs, jd_emb)
        cand_norms = np.linalg.norm(cand_embs, axis=1)
        jd_norm = np.linalg.norm(jd_emb)
        similarities = dot_prod / (cand_norms * jd_norm)
        
    with st.spinner("Step 4: Blending Multi-Signal Metrics & Sorting shortlist..."):
        norm_w_sem = w_sem / w_total if w_total > 0 else 0
        norm_w_exp = w_exp / w_total if w_total > 0 else 0
        norm_w_beh = w_beh / w_total if w_total > 0 else 0
        norm_w_loc = w_loc / w_total if w_total > 0 else 0
        
        final_list = []
        current_date = datetime(2026, 6, 26)
        
        for idx, info in enumerate(candidates_info):
            sem_sim = float(similarities[idx])
            yoe = info['yoe']
            np_days = info['notice_period']
            resp_rate = info['response_rate']
            last_act_str = info['last_active']
            willing_reloc = info['willing_to_relocate']
            open_to_work = info['open_to_work']
            loc = info['location'].lower()
            country = info['country'].lower()
            
            # Experience Score
            if 6.0 <= yoe <= 8.0:
                exp_score = 1.0
            elif 5.0 <= yoe < 6.0:
                exp_score = 0.8 + 0.2 * (yoe - 5.0)
            elif 8.0 < yoe <= 9.0:
                exp_score = 1.0 - 0.2 * (yoe - 8.0)
            elif 4.0 <= yoe < 5.0:
                exp_score = 0.4 + 0.4 * (yoe - 4.0)
            elif 9.0 < yoe <= 10.0:
                exp_score = 0.8 - 0.4 * (yoe - 9.0)
            elif 3.0 <= yoe < 4.0:
                exp_score = 0.1 + 0.3 * (yoe - 3.0)
            elif 10.0 < yoe <= 12.0:
                exp_score = 0.4 - 0.15 * (yoe - 10.0)
            else:
                exp_score = 0.0
                
            # Behavioral Score
            beh_points = 0.0
            if np_days <= 15:
                beh_points += 4.0
            elif np_days <= 30:
                beh_points += 3.0
            elif np_days <= 45:
                beh_points += 2.0
            elif np_days <= 90:
                beh_points += 1.0
            beh_points += resp_rate * 3.0
            
            if last_act_str:
                try:
                    last_act_date = datetime.strptime(last_act_str, "%Y-%m-%d")
                    days_inactive = (current_date - last_act_date).days
                    if days_inactive <= 30:
                        beh_points += 3.0
                    elif days_inactive <= 90:
                        beh_points += 2.0
                    elif days_inactive <= 180:
                        beh_points += 1.0
                except:
                    pass
                    
            if open_to_work:
                beh_points += 2.0
            beh_points += info['interview_completion'] * 3.0
            if info['offer_acceptance'] >= 0.0:
                beh_points += info['offer_acceptance'] * 1.0
            beh_score = beh_points / 16.0
            
            # Location Score
            loc_score = 0.0
            if country == 'india':
                target_cities = ["noida", "pune", "delhi", "gurgaon", "hyderabad", "mumbai", "bengaluru", "bangalore", "chennai"]
                if any(c in loc for c in target_cities):
                    loc_score = 1.0
                elif willing_reloc:
                    loc_score = 0.8
                else:
                    loc_score = 0.5
            else:
                if willing_reloc:
                    loc_score = 0.4
                else:
                    loc_score = 0.0
                    
            final_score = round(norm_w_sem * sem_sim + norm_w_exp * exp_score + norm_w_beh * beh_score + norm_w_loc * loc_score, 4)
            final_list.append((final_score, sem_sim, info))
            
        final_list.sort(key=lambda x: (-x[0], x[2]['candidate_id']))
        st.session_state.ranked_shortlist = final_list[:100]
        st.success("Ranking successfully refreshed!")

# ==========================================
# STATISTICS PANEL (PREMIUM METRIC CARDS)
# ==========================================
if st.session_state.ranked_shortlist is not None:
    st.markdown("### 📊 Matching Funnel Statistics")
    
    total = st.session_state.total_processed
    passed = st.session_state.passed_count
    filtered = total - passed
    
    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card" style="border-left: 4px solid #6366f1;">
            <div class="metric-title">🔍 Candidates Screened</div>
            <div class="metric-value-custom">{total:,}</div>
        </div>
        <div class="metric-card" style="border-left: 4px solid #10b981;">
            <div class="metric-title">✅ Passed Logic Filters</div>
            <div class="metric-value-custom">{passed:,}</div>
        </div>
        <div class="metric-card" style="border-left: 4px solid #ef4444;">
            <div class="metric-title">🛡️ Trapped & Discarded</div>
            <div class="metric-value-custom">{filtered:,}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Filter stats split panels
    col_stat1, col_stat2 = st.columns([2, 1])
    with col_stat1:
        if st.session_state.filter_stats:
            st.markdown("**🛡️ Pre-Screening Trap Breakdown**")
            df_reasons = pd.DataFrame(
                list(st.session_state.filter_stats.items()), 
                columns=["Filter Trigger ID / Contradiction", "Count"]
            ).sort_values("Count", ascending=False)
            st.dataframe(df_reasons, use_container_width=True, hide_index=True)
            
    with col_stat2:
        st.markdown("**💾 Shortlist Export**")
        csv_data = []
        for r_idx, (score, sem_sim, info) in enumerate(st.session_state.ranked_shortlist, 1):
            reasoning = generate_reasoning(info, score, sem_sim)
            csv_data.append({
                "candidate_id": info["candidate_id"],
                "rank": r_idx,
                "score": score,
                "reasoning": reasoning
            })
        df_out = pd.DataFrame(csv_data)
        csv_string = df_out.to_csv(index=False)
        
        st.download_button(
            "📥 Download Shortlist CSV",
            data=csv_string,
            file_name="trio_tech.csv",
            mime="text/csv",
            use_container_width=True
        )
        
    # ==========================================
    # SEARCH BAR & SHORTLIST INTERFACE
    # ==========================================
    st.markdown("---")
    st.markdown("### 🏆 Recruiter Shortlist")
    
    with st.expander("ℹ️ What do the premium tags mean?"):
        if data_option == "Sample Dataset (50 Profiles)":
            st.markdown("""
            *(Demo Mode: Thresholds are slightly loosened for this 50-profile sample so you can see the UI features in action!)*
            - **🚀 Fast Tracker:** Flags candidates with Senior titles but <7 years experience, or who started as an intern.
            - **💎 High Value ROI:** Flags candidates asking for a salary (LPA) less than 4x their YoE, with at least 1 matching skill.
            - **⚠️ Flight Risk:** Flags candidates who have been at their current company for >2 years, or have a short notice period (<=30 days).
            """)
        else:
            st.markdown("""
            *(Production Mode: Extremely strict thresholds are applied for the massive database to highlight only the top 1%)*
            - **🚀 Fast Tracker:** Flags candidates who progressed rapidly from Intern/Junior straight to Senior/Lead/Manager roles.
            - **💎 High Value ROI:** Flags candidates asking for a highly competitive salary (<3x YoE in LPA) despite having 3+ matching technical skills.
            - **⚠️ Flight Risk:** Flags candidates who are highly likely to leave based on a >3 year tenure combined with active 'open to work' signals.
            """)
    
    search_query = st.text_input("🔍 Filter matches by name, current title, or matching skills...", "").strip().lower()
    
    filtered_list = []
    for rank_idx, (score, sem_sim, info) in enumerate(st.session_state.ranked_shortlist, 1):
        name = info['name'].lower()
        title = info['title'].lower()
        skills = " ".join(info['matched_skills']).lower()
        if not search_query or (search_query in name or search_query in title or search_query in skills):
            filtered_list.append((rank_idx, score, sem_sim, info))
            
    if not filtered_list:
        st.info("No candidates match your search query filter.")
    else:
        for r_idx, score, sem_sim, info in filtered_list:
            reasoning = generate_reasoning(info, score, sem_sim)
            
            # Format skills as tags
            skills_html = "".join([f'<span class="badge-premium badge-skills">{s}</span>' for s in info['matched_skills'][:5]])
            
            # Feature: Velocity, ROI, Flight Risk Tags
            velocity_html = '<span class="badge-premium" style="background-color: #f59e0b; color: #fff;">🚀 Fast Tracker</span>' if info.get('velocity') else ""
            roi_html = '<span class="badge-premium" style="background-color: #8b5cf6; color: #fff;">💎 High Value ROI</span>' if info.get('high_roi') else ""
            flight_html = '<span class="badge-premium" style="background-color: #ef4444; color: #fff;">⚠️ Flight Risk</span>' if info.get('flight_risk') else ""
            extra_tags = velocity_html + roi_html + flight_html
            
            # Feature: Blind Audition Mode
            display_name = f"Candidate #{info['candidate_id'][-5:]}" if blind_mode else info['name']
            
            # Behavioral signals logic colors
            np_days = info['notice_period']
            np_dot = "dot-green" if np_days <= 30 else ("dot-orange" if np_days <= 90 else "dot-red")
            
            resp_rate = int(info['response_rate'] * 100)
            resp_dot = "dot-green" if resp_rate >= 80 else ("dot-orange" if resp_rate >= 50 else "dot-red")
            
            last_active = info['last_active']
            
            st.markdown(f"""
<div class="candidate-card-premium">
<div style="display: flex; justify-content: space-between; align-items: start; flex-wrap: wrap; gap: 10px;">
<div>
<div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
<span class="badge-premium badge-rank">Rank {r_idx}</span>
<span class="badge-premium badge-score">Match Score: {score:.4f}</span>
<span class="badge-premium badge-yoe">{info['yoe']} YoE</span>
{extra_tags}
</div>
<h3 style="margin: 12px 0 4px 0; color: var(--text-color); font-size: 22px; font-weight: 700;">{display_name}</h3>
<p style="margin: 0 0 12px 0; color: var(--primary-color); font-weight: 600; font-size: 14.5px;">{info['title']} &bull; <span style="color: var(--text-color); opacity: 0.7; font-weight: normal;">{info['location']}</span></p>
<div style="margin-top: 6px; margin-bottom: 6px;">
{skills_html}
</div>
</div>
<div style="text-align: right; min-width: 120px;">
<span style="font-size: 10px; color: var(--text-color); opacity: 0.7; font-weight: 800; text-transform: uppercase; letter-spacing: 0.5px;">Similarity</span>
<div style="font-size: 24px; font-weight: 800; color: var(--primary-color); margin-top: 4px;">{sem_sim:.4f}</div>
</div>
</div>
<div class="signal-container">
<div class="signal-item">
<span class="signal-dot {np_dot}"></span>
Notice: {np_days} days
</div>
<div class="signal-item">
<span class="signal-dot {resp_dot}"></span>
Response: {resp_rate}%
</div>
<div class="signal-item">
<span class="signal-dot dot-green"></span>
Active: {last_active}
</div>
<div class="signal-item">
<span class="signal-dot dot-green" style="background-color: {'#10b981' if info['willing_to_relocate'] else '#475569'}; box-shadow: 0 0 8px {'#10b981' if info['willing_to_relocate'] else '#475569'};"></span>
Relocating: {'Yes' if info['willing_to_relocate'] else 'No'}
</div>
</div>
<div class="reasoning-box">
<strong>🔍 Fit & Gap Analysis:</strong> {reasoning}
</div>
</div>
""", unsafe_allow_html=True)
            
            with st.expander(f"📝 Full Profile & Career Timeline - {display_name}"):
                col_ex1, col_ex2 = st.columns(2)
                with col_ex1:
                    st.markdown("**Headline & Background:**")
                    st.write(info['headline'])
                    st.markdown("**Summary:**")
                    st.write(info['summary'])
                with col_ex2:
                    st.markdown("**Logistics & Rates:**")
                    st.write(f"- Stated Notice Period: {info['notice_period']} Days")
                    st.write(f"- Recruiter Response Rate: {resp_rate}%")
                    st.write(f"- Interview Attendance Rate: {int(info['interview_completion']*100)}%")
                    st.write(f"- Relocation Willingness: {'Yes' if info['willing_to_relocate'] else 'No'}")
                    
                st.markdown("**Career History:**")
                for job in info['career_history']:
                    st.markdown(f"**{job['title']}** at *{job['company']}* ({job['start_date']} to {job['end_date'] or 'Present'})")
                    st.caption(job['description'])
                    st.markdown("---")
                
                # Feature: Draft Email
                if st.button(f"🪄 Draft Outreach Email for {display_name}", key=f"email_{info['candidate_id']}"):
                    first_name = display_name.split(' ')[0] if not blind_mode else "Candidate"
                    skills_str = ", ".join(info['matched_skills'][:3]) if info['matched_skills'] else "your technical skills"
                    email_draft = f"Subject: Exciting Opportunity for a {info['title']}\n\nHi {first_name},\n\nI was incredibly impressed by your background as a {info['title']} and your strong expertise in {skills_str}. Given your {info['yoe']} years of experience, I think you'd be a fantastic fit for a Senior role we are currently hiring for.\n\nI saw you are currently based in {info['location']}. Would you be open to a quick 10-minute chat this week to discuss?\n\nBest regards,\n[Your Name]"
                    st.text_area("✨ Generated Draft:", value=email_draft, height=220, key=f"draft_text_{info['candidate_id']}")
else:
    st.info("Initiate matching by clicking the search button above.")
