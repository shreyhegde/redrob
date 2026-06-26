import json
import os
import csv
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer

# Define file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "[PUB] India_runs_data_and_ai_challenge", "India_runs_data_and_ai_challenge", "candidates.jsonl")
OUTPUT_PATH = os.path.join(BASE_DIR, "team_antigravity.csv")
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

KEYWORDS = ["ranking", "search", "retrieval", "vector", "embedding", "recommendation", "nlp", "llm", "rag", "eval", "benchmark", "production", "scale", "infrastructure", "pipeline", "distributed", "index", "re-rank", "hybrid search", "bm25", "fine-tune"]

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

def check_filters_and_compute_heuristic(c):
    """
    Stage 0 & 1: Filters out honeypots/exclusions and computes a fast heuristic score.
    Returns (heuristic_score, candidate_info_dict) if passed, else (None, reason).
    """
    cid = c['candidate_id']
    profile = c['profile']
    history = c['career_history']
    edu = c['education']
    skills = c['skills']
    signals = c['redrob_signals']
    yoe = profile.get('years_of_experience', 0)
    
    current_date = datetime(2026, 6, 26)

    # ==========================================
    # 0. STRICT HONEYPOT FILTERS
    # ==========================================
    
    # A. expected salary min > max
    sal = signals.get('expected_salary_range_inr_lpa', {})
    if sal.get('min', 0) > sal.get('max', 0):
        return None, "salary_min_gt_max"
        
    # B. certifications in the future
    if any(cert.get('year', 0) > 2026 for cert in c.get('certifications', [])):
        return None, "cert_future"
        
    # C. role duration > yoe + 0.5
    if any(r.get('duration_months', 0) / 12.0 > yoe + 0.5 for r in history):
        return None, "role_dur_gt_yoe"
        
    # D. assessment score for a skill not in the profile skills list
    skill_names = {s['name'].lower() for s in skills}
    if any(sk.lower() not in skill_names for sk in signals.get('skill_assessment_scores', {})):
        return None, "assessment_skill_not_in_profile_skills"
        
    # E. career start date is way before college end date (logical impossibility)
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
                    # Full time role starting before university
                    if s_year < earliest_edu_end - 6 and r.get('duration_months', 0) > 36:
                        has_pre_edu_career = True
                        break
                except:
                    pass
        if has_pre_edu_career:
            return None, "career_start_pre_edu"

    # F. Impossible Job Durations relative to calendar dates
    for r in history:
        sd = parse_date(r.get('start_date'))
        ed = parse_date(r.get('end_date')) or current_date
        if sd:
            cal_months = (ed.year - sd.year) * 12 + (ed.month - sd.month)
            if r.get('duration_months', 0) > cal_months + 2:
                return None, "job_duration_impossible"

    # G. Expert / Advanced skill but zero or negative months
    for s in skills:
        prof = s.get('proficiency', '')
        dur = s.get('duration_months', 0)
        if prof in ('expert', 'advanced') and dur <= 0:
            return None, "skills_impossible"

    # ==========================================
    # 1. EXCLUSION FILTERS
    # ==========================================

    # A. Blacklisted Title
    current_title = profile.get('current_title', '').lower()
    if current_title in BLACKLISTED_TITLES:
        return None, "blacklisted_title"
    
    if any(term in current_title for term in ["marketing", "sales", "hr manager", "accountant", "designer", "operations manager", "support", "civil engineer"]):
        return None, "blacklisted_title_contains"

    # B. Entire career is in IT services/consulting firms
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

    # C. Computer vision / speech / robotics primary without NLP/IR exposure
    has_cv = any(any(cvs in s['name'].lower() for cvs in CV_SKILLS) for s in skills)
    has_nlp = any(any(nlps in s['name'].lower() for nlps in NLP_SKILLS) for s in skills)
    if has_cv and not has_nlp:
        return None, "cv_only_no_nlp"

    # D. Academic research only
    all_research = True
    for r in history:
        title = r.get('title', '').lower()
        if not any(res in title for res in ["researcher", "research assistant", "phd student", "intern", "academic"]):
            all_research = False
            break
    if all_research and len(history) > 0:
        return None, "research_only"

    # E. Excessive job switching (average tenure under 1.5 years)
    if len(history) >= 3:
        avg_tenure = (yoe * 12.0) / len(history)
        if avg_tenure < 18.0:
            return None, "frequent_job_switching"

    # ==========================================
    # 2. FAST HEURISTIC SCORE (for selecting top 2000)
    # ==========================================
    heuristic_score = 0.0

    # A. Role Title Match
    if any(t in current_title for t in TARGET_TITLES):
        heuristic_score += 20.0
    elif any(t in current_title for t in ["software engineer", "backend engineer", "data engineer"]):
        heuristic_score += 10.0

    # B. Fast Skills Score
    skills_match_count = 0
    matched_skills_list = []
    for s in skills:
        sname = s['name'].lower()
        proficiency = s.get('proficiency', 'beginner')
        prof_mult = {"expert": 1.0, "advanced": 0.8, "intermediate": 0.6, "beginner": 0.4}.get(proficiency, 0.4)
        
        s_weight = 0
        for sw_name, w in SKILL_WEIGHTS.items():
            if sw_name in sname:
                s_weight = max(s_weight, w)
                
        if s_weight > 0:
            heuristic_score += s_weight * prof_mult
            skills_match_count += 1
            matched_skills_list.append(s['name'])

    # C. Experience Score
    if 6.0 <= yoe <= 8.0:
        heuristic_score += 15.0
    elif 5.0 <= yoe < 6.0 or 8.0 < yoe <= 9.0:
        heuristic_score += 10.0
    elif 4.0 <= yoe < 5.0 or 9.0 < yoe <= 10.0:
        heuristic_score += 5.0

    # D. Behavioral Signals
    np_days = signals.get('notice_period_days', 90)
    if np_days <= 15:
        heuristic_score += 5.0
    elif np_days <= 30:
        heuristic_score += 4.0
    elif np_days <= 45:
        heuristic_score += 2.0
        
    resp_rate = signals.get('recruiter_response_rate', 0.0)
    heuristic_score += resp_rate * 5.0

    if signals.get('open_to_work_flag'):
        heuristic_score += 3.0

    # Location
    country = profile.get('country', '').lower()
    loc = profile.get('location', '').lower()
    if country == 'india':
        heuristic_score += 5.0
        target_cities = ["noida", "pune", "delhi", "gurgaon", "hyderabad", "mumbai", "bengaluru", "bangalore", "chennai"]
        if any(c in loc for c in target_cities):
            heuristic_score += 5.0

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
        'career_history': history
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
    
    # 1. Experience & Title
    exp_phrase = f"{yoe} YoE as a {title}"
    
    # 2. Key matching skills
    core_skills = [s for s in skills if s.lower() in ["vector search", "pinecone", "milvus", "qdrant", "weaviate", "faiss", "elasticsearch", "ndcg", "mrr", "map", "information retrieval", "ranking", "llm", "fine-tuning", "rag"]]
    if core_skills:
        skills_phrase = f"expert alignment in {', '.join(core_skills[:3])}"
    else:
        skills_phrase = f"solid background in {', '.join(skills[:3]) if skills else 'applied ML'}"
        
    # 3. Location/Relocation
    in_target_city = any(c in location.lower() for c in ["noida", "pune", "delhi", "gurgaon", "hyderabad", "mumbai", "bengaluru", "bangalore", "chennai"])
    if in_target_city:
        loc_phrase = f"based in {location}"
    elif willing_reloc:
        loc_phrase = f"willing to relocate to Pune/Noida"
    else:
        loc_phrase = f"based in {location}"
        
    # 4. Behavioral & Notice Period
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
        
    # 5. Gaps / Concerns (crucial for Stage 4 manual review)
    concerns = []
    if yoe < 5.0:
        concerns.append("experience is slightly below the 5-9 year target")
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
        
    # Put it all together dynamically using candidate hash to introduce variation
    h = hash(cid) % 3
    if h == 0:
        reasoning = f"Strong candidate with {exp_phrase}, {loc_phrase}. Proven fit with {skills_phrase}; {avail_phrase} and {eng_phrase}.{concern_text}"
    elif h == 1:
        reasoning = f"{exp_phrase} showing {skills_phrase}. Located in {location}, {eng_phrase} and {avail_phrase}.{concern_text}"
    else:
        reasoning = f"Excellent technical alignment with {skills_phrase} across {exp_phrase}. {avail_phrase}, {loc_phrase}, showing {eng_phrase}.{concern_text}"
        
    return reasoning

def main():
    print("Initiating upgraded candidate screening and ranking...")
    
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset not found at {DATASET_PATH}")
        return
        
    # ============================================================================
    # STAGE 1: Fast filtering & Heuristic Screening
    # ============================================================================
    print("Reading candidate pool and running Stage 1 filtering...")
    candidates_passed = []
    count = 0
    
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            count += 1
            c = json.loads(line)
            h_score, info = check_filters_and_compute_heuristic(c)
            if h_score is not None:
                candidates_passed.append((h_score, info))
                
            if count % 20000 == 0:
                print(f"Processed {count} candidates...")
                
    print(f"Total candidates processed: {count}")
    print(f"Candidates passing initial filters: {len(candidates_passed)}")
    
    # Sort candidates by heuristic score descending, keep top 2000 for semantic re-ranking
    candidates_passed.sort(key=lambda x: -x[0])
    top_candidates = candidates_passed[:2000]
    print(f"Selected top {len(top_candidates)} candidates for semantic re-ranking.")
    
    # ============================================================================
    # STAGE 2: Semantic Similarity
    # ============================================================================
    print("Loading local SentenceTransformer model...")
    model_start = datetime.now()
    if not os.path.exists(MODEL_DIR):
         print(f"Error: Local model directory not found at {MODEL_DIR}")
         return
    model = SentenceTransformer(MODEL_DIR)
    print(f"Model loaded in {(datetime.now() - model_start).total_seconds():.2f} seconds.")
    
    # Define detailed Job Description query representation
    jd_query = (
        "Senior AI Engineer with production experience in embeddings-based retrieval systems, "
        "vector databases like Pinecone, Milvus, Qdrant, Weaviate, FAISS, Elasticsearch, or OpenSearch, "
        "hybrid search infrastructure, and ranking evaluation metrics like NDCG, MRR, and MAP. "
        "Production deployment of search, retrieval, and learning-to-rank systems, offline evaluation "
        "and A/B testing, Python, PyTorch."
    )
    
    print("Encoding Job Description query...")
    jd_embedding = model.encode(jd_query)
    
    print("Building semantic profiles and encoding candidates...")
    candidate_texts = []
    candidates_info = []
    
    for h_score, info in top_candidates:
        # Construct a rich text representation of the profile and recent career history
        profile_text = f"Headline: {info['headline']}. Summary: {info['summary']}. Current Title: {info['title']}."
        history_parts = []
        for job in info['career_history'][:3]:
            history_parts.append(f"Role: {job['title']} at {job['company']}. Description: {job['description']}")
        history_text = " ".join(history_parts)
        full_candidate_text = f"{profile_text} {history_text}"
        
        candidate_texts.append(full_candidate_text)
        candidates_info.append(info)
        
    candidate_embeddings = model.encode(candidate_texts, batch_size=64, show_progress_bar=False)
    
    # Compute cosine similarities manually using numpy to run offline and fast
    print("Computing semantic similarity scores...")
    dot_products = np.dot(candidate_embeddings, jd_embedding)
    candidate_norms = np.linalg.norm(candidate_embeddings, axis=1)
    jd_norm = np.linalg.norm(jd_embedding)
    similarities = dot_products / (candidate_norms * jd_norm)
    
    # ============================================================================
    # STAGE 3: Multi-Signal Score Blending
    # ============================================================================
    print("Blending semantic, experience, behavioral, and location scores...")
    final_results = []
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
        
        # A. Experience alignment score (0.0 to 1.0)
        # Target: 5-9 years, ideal: 6-8 years
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
            
        # B. Behavioral score (0.0 to 1.0)
        beh_points = 0.0
        
        # Notice period (max 4.0)
        if np_days <= 15:
            beh_points += 4.0
        elif np_days <= 30:
            beh_points += 3.0
        elif np_days <= 45:
            beh_points += 2.0
        elif np_days <= 90:
            beh_points += 1.0
            
        # Response rate (max 3.0)
        beh_points += resp_rate * 3.0
        
        # Last active date (max 3.0)
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
                
        # Open to work flag (max 2.0)
        if open_to_work:
            beh_points += 2.0
            
        # Interview completion rate (max 3.0)
        beh_points += info['interview_completion'] * 3.0
        
        # Offer acceptance rate (max 1.0)
        if info['offer_acceptance'] >= 0.0:
            beh_points += info['offer_acceptance'] * 1.0
            
        beh_score = beh_points / 16.0
        
        # C. Location score (0.0 to 1.0)
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
                
        # D. Blend scores
        # 50% semantic similarity, 15% experience curve, 25% behavioral active signals, 10% location logistics
        final_score = round(0.50 * sem_sim + 0.15 * exp_score + 0.25 * beh_score + 0.10 * loc_score, 4)
        
        final_results.append((final_score, sem_sim, info))
        
    # Sort candidates: score descending, then candidate_id ascending (deterministic tie-breaker)
    print("Sorting and ranking candidates...")
    final_results.sort(key=lambda x: (-x[0], x[2]['candidate_id']))
    
    # Extract top 100
    top_100 = final_results[:100]
    
    print("Writing ranked shortlist to CSV...")
    with open(OUTPUT_PATH, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        
        for rank_idx, (score, sem_sim, info) in enumerate(top_100, 1):
            cid = info['candidate_id']
            reasoning = generate_reasoning(info, score, sem_sim)
            writer.writerow([cid, rank_idx, score, reasoning])
            
    print(f"Shortlist completed and saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
