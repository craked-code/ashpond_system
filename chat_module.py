import json
import config
import numpy as np
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_client = None
_vectorizer = None
_qa_vectors = None

SYSTEM_PROMPT = """You are CoalWatch, an AI risk analyst for coal ash pond breach early warning in India. You are deployed during the 2024 monsoon season.

You monitor 10 ash ponds across two coal belts:
- Singrauli, Madhya Pradesh: Sasan UMPP (P001, P002), Vindhyachal STPS (P003, P004), Rihand STPS (P005)
- Korba, Chhattisgarh: NTPC Korba (P006, P007), CSEB Korba West (P008), Korba East TPS (P009), Hasdeo TPS (P010)

RISK FRAMEWORK:
- DII (Dyke Instability Index): primary indicator, range 0–1. CRITICAL > 0.70, ELEVATED 0.40–0.70, STABLE < 0.40
- DII is computed from: NDWI (seepage proxy, weight 0.28), NDVI stress (vegetation die-off, weight 0.24), slope (weight 0.19), rainfall forecast (weight 0.15), historical breach proximity (weight 0.09), SAR backscatter (weight 0.05)
- breach_probability: Cox Proportional Hazards survival model, calibrated on 12 real Indian ash pond disaster records including Sasan 2020, NTPC Rihand, Korba incidents from CAG and NGT records. Outputs P(breach within 30 days)
- DII is the primary indicator. breach_probability is secondary and may be low even for CRITICAL ponds due to small training set
- Bayesian uncertainty: DII shown with 95% confidence interval from 1000 bootstrap samples
- Anomalous ponds: Mahalanobis distance > 75th percentile of fleet — statistically unusual risk factor combination even if DII appears moderate
- SAR backscatter from Sentinel-1 radar penetrates monsoon cloud cover when Sentinel-2 optical is blinded

HISTORICAL CONTEXT:
- April 20 2020: Sasan UMPP dyke breached. 3 killed. Slurry reached Rihand reservoir. Our retrospective model shows DII crossed 0.70 on April 7 — 13 days before breach.
- This system is designed to provide that 13-day window operationally.

BEHAVIOUR:
- Always call tools when asked about specific ponds, simulations, or rankings. Never invent numbers.
- Lead every answer with the key number, then interpret it in one sentence.
- Be direct. Judges and field engineers need fast, actionable answers.
- If asked to compare ponds, rank by DII first, then breach_probability.
- If asked what to do, recommend inspection for CRITICAL, increased monitoring for ELEVATED, routine checks for STABLE."""

QA_BANK = {
    "which pond is most dangerous": None,
    "which pond should be inspected first": None,
    "what is the dii at sasan": None,
    "why is sasan pond critical": None,
    "what happens if rain increases at korba": None,
    "how many ponds are critical": None,
    "how many ponds are elevated": None,
    "what is the breach probability at sasan": None,
    "which ponds are anomalous": None,
    "what does anomalous mean": None,
    "compare sasan and korba": None,
    "what is ndwi": None,
    "what is sar data": None,
    "explain the dii formula": None,
    "what happened in 2020 at sasan": None,
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_pond_status",
            "description": "Get DII score, breach probability, anomaly flag for a pond",
            "parameters": {
                "type": "object",
                "properties": {
                    "pond_name": {"type": "string", "description": "Partial or full pond name"}
                },
                "required": ["pond_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "simulate_rainfall",
            "description": "Calculate new DII and breach probability if extra rainfall occurs at a pond",
            "parameters": {
                "type": "object",
                "properties": {
                    "pond_name": {"type": "string"},
                    "extra_mm":  {"type": "number", "description": "Extra rainfall in mm"}
                },
                "required": ["pond_name", "extra_mm"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_top_risk_ponds",
            "description": "Return the top N ponds ranked by DII",
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {"type": "integer", "default": 3}
                },
                "required": ["n"]
            }
        }
    },
]

def _get_client():
    global _client
    if _client is None:
        _client = Groq(api_key=config.GROQ_API_KEY)
    return _client

def _build_semantic_index():
    global _vectorizer, _qa_vectors
    _vectorizer = TfidfVectorizer()
    _qa_vectors = _vectorizer.fit_transform(list(QA_BANK.keys()))

def _execute_tool(tool_name, args, df):
    from data_loader import get_pond_by_name_fuzzy
    
    if tool_name == "get_pond_status":
        row = get_pond_by_name_fuzzy(args.get("pond_name", ""), df)
        if row is None:
            return "Pond not found."
        return json.dumps({
            "pond": row['pond_name'],
            "dii": float(row['dii_score']),
            "breach_probability": float(row.get('breach_probability', 0)),
            "risk_category": row['risk_category'],
            "anomalous": bool(row.get('profile_anomaly', False)),
        })

    elif tool_name == "simulate_rainfall":
        row = get_pond_by_name_fuzzy(args.get("pond_name", ""), df)
        if row is None:
            return "Pond not found."
        extra = float(args.get("extra_mm", 0))
        new_dii = min(float(row['dii_score']) + (extra / 500), 1.0)
        new_prob = min(float(row.get('breach_probability', 0)) + (extra / 400), 1.0)
        new_cat = "CRITICAL" if new_dii > 0.70 else "ELEVATED" if new_dii > 0.40 else "STABLE"
        return json.dumps({
            "pond": row['pond_name'],
            "extra_rain_mm": extra,
            "simulated_dii": round(new_dii, 2),
            "simulated_probability": round(new_prob, 2),
            "new_category": new_cat,
        })

    elif tool_name == "get_top_risk_ponds":
        n = int(args.get("n", 3))
        top = df.sort_values('dii_score', ascending=False).head(n)
        return json.dumps([{
            "pond": r['pond_name'],
            "dii": float(r['dii_score']),
            "breach_probability": float(r.get('breach_probability', 0)),
            "risk_category": r['risk_category']
        } for _, r in top.iterrows()])

    return "Tool not found."

def ask_assistant(user_message, conversation_history, df):
    try:
        client = _get_client()
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        for msg in conversation_history[-6:]:
            if msg["role"] in ("user", "assistant"):
                messages.append({"role": msg["role"], "content": msg["content"]})
        
        messages.append({"role": "user", "content": user_message})

        resp = client.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=400,
            temperature=0.2,
        )
        msg = resp.choices[0].message

        if msg.tool_calls:
            tool_results = []
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                result = _execute_tool(tc.function.name, args, df)
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
            
            messages.append({"role": "assistant", "tool_calls": msg.tool_calls})
            messages.extend(tool_results)
            
            final = client.chat.completions.create(
                model=config.GROQ_MODEL,
                messages=messages,
                max_tokens=300,
                temperature=0.2,
            )
            return final.choices[0].message.content

        return msg.content

    except Exception as e:
        # Gracefully handle API failures with vector math matching
        return _semantic_fallback(user_message, df)

def _semantic_fallback(question, df):
    global _vectorizer, _qa_vectors
    if _vectorizer is None:
        _build_semantic_index()

    critical = df[df['risk_category'] == 'CRITICAL'].sort_values('dii_score', ascending=False)
    top = critical.iloc[0] if len(critical) > 0 else df.iloc[0]

    # Pre-calculated answers grounded in current dataset metrics
    answers = {
        "which pond is most dangerous": 
            f"{top['pond_name']} is at top risk tier. DII: {top['dii_score']:.2f}, Breach Prob: {int(top.get('breach_probability',0)*100)}%. System flag: {top.get('top_risk_factor', 'Seepage Activity')}.",
        "which pond should be inspected first": 
            f"Deployment priority should be assigned to {top['pond_name']}. Core instability markers are active with DII {top['dii_score']:.2f}.",
        "how many ponds are critical": 
            f"Analytics scan reveals {len(df[df.risk_category=='CRITICAL'])} CRITICAL ponds, {len(df[df.risk_category=='ELEVATED'])} ELEVATED, and {len(df[df.risk_category=='STABLE'])} STABLE records.",
        "which ponds are anomalous": 
            f"Anomalous profiles discovered at: {', '.join(df[df['profile_anomaly']==True]['pond_name'].tolist()) if 'profile_anomaly' in df.columns else 'None detected'}.",
        "what happened in 2020 at sasan": 
            "On April 20, 2020, the Sasan UMPP dyke ruptured. Retrospective analysis proves the structure's structural index crossed critical parameters on April 7, validating a 13-day early warning buffer.",
    }

    # Vectorize input query and check against target banking queries
    q_vec = _vectorizer.transform([question.lower()])
    sims = cosine_similarity(q_vec, _qa_vectors)[0]
    best_match_idx = sims.argmax()

    if sims[best_match_idx] > 0.25:
        best_key = list(QA_BANK.keys())[best_match_idx]
        if best_key in answers:
            return answers[best_key]

    # Quick fuzzy lookup directly within question text structure
    from data_loader import get_pond_by_name_fuzzy
    row = get_pond_by_name_fuzzy(question, df)
    if row is not None:
        return f"{row['pond_name']} Status Summary: DII {row['dii_score']:.2f} [{row['risk_category']}]. P(Breach)={int(row.get('breach_probability',0)*100)}%."

    return "System running on semantic edge fallbacks. Please specify pond target names directly (e.g., 'Sasan' or 'Korba')."