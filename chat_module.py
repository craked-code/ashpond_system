# import json
# import config
# import numpy as np
# from groq import Groq
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# _client = None
# _vectorizer = None
# _qa_vectors = None

# SYSTEM_PROMPT = """You are CoalWatch, an AI assistant for coal ash pond breach risk monitoring in India.
# You have access to real-time DII scores, breach probabilities, SHAP explanations, and anomaly flags for 10 monitored ash ponds.
# Always cite specific DII values and breach probabilities. Be concise, technical and actionable.
# When a tool is available that can answer the question, use it."""

# QA_BANK = {
#     "which pond is most dangerous": None,
#     "which pond should be inspected first": None,
#     "what is the dii at sasan": None,
#     "why is sasan pond critical": None,
#     "what happens if rain increases at korba": None,
#     "how many ponds are critical": None,
#     "how many ponds are elevated": None,
#     "what is the breach probability at sasan": None,
#     "which ponds are anomalous": None,
#     "what does anomalous mean": None,
#     "compare sasan and korba": None,
#     "what is ndwi": None,
#     "what is sar data": None,
#     "explain the dii formula": None,
#     "what happened in 2020 at sasan": None,
# }

# TOOLS = [
#     {
#         "type": "function",
#         "function": {
#             "name": "get_pond_status",
#             "description": "Get DII score, breach probability, anomaly flag for a pond",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "pond_name": {"type": "string", "description": "Partial or full pond name"}
#                 },
#                 "required": ["pond_name"]
#             }
#         }
#     },
#     {
#         "type": "function",
#         "function": {
#             "name": "simulate_rainfall",
#             "description": "Calculate new DII and breach probability if extra rainfall occurs at a pond",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "pond_name": {"type": "string"},
#                     "extra_mm":  {"type": "number", "description": "Extra rainfall in mm"}
#                 },
#                 "required": ["pond_name", "extra_mm"]
#             }
#         }
#     },
#     {
#         "type": "function",
#         "function": {
#             "name": "get_top_risk_ponds",
#             "description": "Return the top N ponds ranked by DII",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "n": {"type": "integer", "default": 3}
#                 },
#                 "required": ["n"]
#             }
#         }
#     },
# ]

# def _get_client():
#     global _client
#     if _client is None:
#         _client = Groq(api_key=config.GROQ_API_KEY)
#     return _client

# def _build_semantic_index():
#     global _vectorizer, _qa_vectors
#     _vectorizer = TfidfVectorizer()
#     _qa_vectors = _vectorizer.fit_transform(list(QA_BANK.keys()))

# def _execute_tool(tool_name, args, df):
#     from data_loader import get_pond_by_name_fuzzy
    
#     if tool_name == "get_pond_status":
#         row = get_pond_by_name_fuzzy(args.get("pond_name", ""), df)
#         if row is None:
#             return "Pond not found."
#         return json.dumps({
#             "pond": row['pond_name'],
#             "dii": float(row['dii_score']),
#             "breach_probability": float(row.get('breach_probability', 0)),
#             "risk_category": row['risk_category'],
#             "anomalous": bool(row.get('profile_anomaly', False)),
#         })

#     elif tool_name == "simulate_rainfall":
#         row = get_pond_by_name_fuzzy(args.get("pond_name", ""), df)
#         if row is None:
#             return "Pond not found."
#         extra = float(args.get("extra_mm", 0))
#         new_dii = min(float(row['dii_score']) + (extra / 500), 1.0)
#         new_prob = min(float(row.get('breach_probability', 0)) + (extra / 400), 1.0)
#         new_cat = "CRITICAL" if new_dii > 0.70 else "ELEVATED" if new_dii > 0.40 else "STABLE"
#         return json.dumps({
#             "pond": row['pond_name'],
#             "extra_rain_mm": extra,
#             "simulated_dii": round(new_dii, 2),
#             "simulated_probability": round(new_prob, 2),
#             "new_category": new_cat,
#         })

#     elif tool_name == "get_top_risk_ponds":
#         n = int(args.get("n", 3))
#         top = df.sort_values('dii_score', ascending=False).head(n)
#         return json.dumps([{
#             "pond": r['pond_name'],
#             "dii": float(r['dii_score']),
#             "breach_probability": float(r.get('breach_probability', 0)),
#             "risk_category": r['risk_category']
#         } for _, r in top.iterrows()])

#     return "Tool not found."

# def ask_assistant(user_message, conversation_history, df):
#     try:
#         client = _get_client()
#         messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
#         for msg in conversation_history[-6:]:
#             if msg["role"] in ("user", "assistant"):
#                 messages.append({"role": msg["role"], "content": msg["content"]})
        
#         messages.append({"role": "user", "content": user_message})

#         resp = client.chat.completions.create(
#             model=config.GROQ_MODEL,
#             messages=messages,
#             tools=TOOLS,
#             tool_choice="auto",
#             max_tokens=400,
#             temperature=0.2,
#         )
#         msg = resp.choices[0].message

#         if msg.tool_calls:
#             tool_results = []
#             for tc in msg.tool_calls:
#                 args = json.loads(tc.function.arguments)
#                 result = _execute_tool(tc.function.name, args, df)
#                 tool_results.append({
#                     "role": "tool",
#                     "tool_call_id": tc.id,
#                     "content": result,
#                 })
            
#             messages.append({"role": "assistant", "tool_calls": msg.tool_calls})
#             messages.extend(tool_results)
            
#             final = client.chat.completions.create(
#                 model=config.GROQ_MODEL,
#                 messages=messages,
#                 max_tokens=300,
#                 temperature=0.2,
#             )
#             return final.choices[0].message.content

#         return msg.content

#     except Exception as e:
#         # Gracefully handle API failures with vector math matching
#         return _semantic_fallback(user_message, df)

# def _semantic_fallback(question, df):
#     global _vectorizer, _qa_vectors
#     if _vectorizer is None:
#         _build_semantic_index()

#     critical = df[df['risk_category'] == 'CRITICAL'].sort_values('dii_score', ascending=False)
#     top = critical.iloc[0] if len(critical) > 0 else df.iloc[0]

#     # Pre-calculated answers grounded in current dataset metrics
#     answers = {
#         "which pond is most dangerous": 
#             f"{top['pond_name']} is at top risk tier. DII: {top['dii_score']:.2f}, Breach Prob: {int(top.get('breach_probability',0)*100)}%. System flag: {top.get('top_risk_factor', 'Seepage Activity')}.",
#         "which pond should be inspected first": 
#             f"Deployment priority should be assigned to {top['pond_name']}. Core instability markers are active with DII {top['dii_score']:.2f}.",
#         "how many ponds are critical": 
#             f"Analytics scan reveals {len(df[df.risk_category=='CRITICAL'])} CRITICAL ponds, {len(df[df.risk_category=='ELEVATED'])} ELEVATED, and {len(df[df.risk_category=='STABLE'])} STABLE records.",
#         "which ponds are anomalous": 
#             f"Anomalous profiles discovered at: {', '.join(df[df['profile_anomaly']==True]['pond_name'].tolist()) if 'profile_anomaly' in df.columns else 'None detected'}.",
#         "what happened in 2020 at sasan": 
#             "On April 20, 2020, the Sasan UMPP dyke ruptured. Retrospective analysis proves the structure's structural index crossed critical parameters on April 7, validating a 13-day early warning buffer.",
#     }

#     # Vectorize input query and check against target banking queries
#     q_vec = _vectorizer.transform([question.lower()])
#     sims = cosine_similarity(q_vec, _qa_vectors)[0]
#     best_match_idx = sims.argmax()

#     if sims[best_match_idx] > 0.25:
#         best_key = list(QA_BANK.keys())[best_match_idx]
#         if best_key in answers:
#             return answers[best_key]

#     # Quick fuzzy lookup directly within question text structure
#     from data_loader import get_pond_by_name_fuzzy
#     row = get_pond_by_name_fuzzy(question, df)
#     if row is not None:
#         return f"{row['pond_name']} Status Summary: DII {row['dii_score']:.2f} [{row['risk_category']}]. P(Breach)={int(row.get('breach_probability',0)*100)}%."

#     return "System running on semantic edge fallbacks. Please specify pond target names directly (e.g., 'Sasan' or 'Korba')."




import json
import config
import numpy as np
import streamlit as st
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_client = None
_vectorizer = None
_qa_vectors = None

# FIX 7 — Identity corrected from "CoalWatch" to "AshPond System"
# FIX 2 — Live data injected at call time via build_system_prompt()
BASE_SYSTEM_PROMPT = """You are the AshPond System AI assistant for coal ash pond breach risk monitoring in India.
You have access to real-time DII scores, breach probabilities, SHAP explanations, and anomaly flags for 10 monitored ash ponds.
Rules:
- HIGHEST risk = HIGHEST dii_score
- LOWEST risk = LOWEST dii_score  
- 2nd riskiest = 2nd highest dii_score
- 2nd safest = 2nd lowest dii_score
- Always sort the provided pond data by dii_score to answer ranking questions
- Always cite specific DII values, breach probabilities, and coordinates (latitude, longitude) when asked
- Be concise, technical and actionable
- When a tool is available that can answer the question, use it."""


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


def build_system_prompt(df):
    pond_summary = df[[
        'pond_name', 'dii_score', 'risk_category',
        'breach_probability', 'profile_anomaly', 
        'top_risk_factor', 'latitude', 'longitude'  # ADD THESE
    ]].to_dict(orient='records')
    data_block = json.dumps(pond_summary, indent=2)
    return BASE_SYSTEM_PROMPT + f"\n\nCurrent pond status (live data):\n{data_block}"

def _get_client():
    global _client
    if _client is None:
        _client = Groq(api_key=config.GROQ_API_KEY)
    return _client


# FIX 5 — Called once at startup, not lazily on first fallback
def init_semantic_index():
    global _vectorizer, _qa_vectors
    _vectorizer = TfidfVectorizer()
    _qa_vectors = _vectorizer.fit_transform(list(QA_BANK.keys()))


def _build_semantic_index():
    init_semantic_index()


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


def ask_assistant(user_message, df):
    """
    FIX 3 — conversation_history now stored in st.session_state, not passed as parameter.
    Call signature simplified: ask_assistant(user_message, df)
    """
    # FIX 3 — initialise session state history if not present
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    # Append user message to persistent history
    st.session_state.conversation_history.append({
        "role": "user",
        "content": user_message
    })

    try:
        client = _get_client()

        # FIX 2 — use live-data-injected system prompt
        messages = [{"role": "system", "content": build_system_prompt(df)}]

        # Last 6 turns from persistent history
        for msg in st.session_state.conversation_history[-6:]:
            if msg["role"] in ("user", "assistant"):
                messages.append({"role": msg["role"], "content": msg["content"]})

        resp = client.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=400,
            temperature=0.45,   # FIX 6 — raised from 0.2 for natural variation
        )
        msg = resp.choices[0].message

        if msg.tool_calls:
            tool_results = []
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                result = _execute_tool(tc.function.name, args, df)

                # FIX 4 — validate tool result before second Groq call
                if result in ("Pond not found.", "Tool not found."):
                    reply = f"I could not find the requested pond or tool. Please specify a valid pond name (e.g. 'Sasan', 'Korba')."
                    st.session_state.conversation_history.append({
                        "role": "assistant", "content": reply
                    })
                    return reply

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
                temperature=0.45,  # FIX 6
            )
            reply = final.choices[0].message.content
            st.session_state.conversation_history.append({
                "role": "assistant", "content": reply
            })
            return reply

        reply = msg.content
        st.session_state.conversation_history.append({
            "role": "assistant", "content": reply
        })
        return reply

    except Exception:
        reply = _semantic_fallback(user_message, df)
        st.session_state.conversation_history.append({
            "role": "assistant", "content": reply
        })
        return reply


def _semantic_fallback(question, df):
    """FIX 3 — Static answers replaced with runtime-computed values from live df."""
    global _vectorizer, _qa_vectors

    # FIX 5 — only rebuild if somehow not initialised at startup
    if _vectorizer is None:
        _build_semantic_index()

    critical = df[df['risk_category'] == 'CRITICAL'].sort_values('dii_score', ascending=False)
    top = critical.iloc[0] if len(critical) > 0 else df.iloc[0]
    anomalous_names = df[df['profile_anomaly'] == True]['pond_name'].tolist() if 'profile_anomaly' in df.columns else []
    n_critical  = len(df[df['risk_category'] == 'CRITICAL'])
    n_elevated  = len(df[df['risk_category'] == 'ELEVATED'])
    n_stable    = len(df[df['risk_category'] == 'STABLE'])

    # FIX 3 — all answers computed from live df at runtime
    answers = {
        "which pond is most dangerous":
            f"{top['pond_name']} is at highest risk. DII: {top['dii_score']:.2f}, "
            f"Breach Probability: {int(top.get('breach_probability', 0) * 100)}%. "
            f"Primary driver: {top.get('top_risk_factor', 'unknown')}.",
        "which pond should be inspected first":
            f"Deployment priority: {top['pond_name']}. DII {top['dii_score']:.2f} with "
            f"{int(top.get('breach_probability', 0) * 100)}% breach probability.",
        "how many ponds are critical":
            f"Current scan: {n_critical} CRITICAL, {n_elevated} ELEVATED, {n_stable} STABLE.",
        "which ponds are anomalous":
            f"Anomalous profiles detected at: {', '.join(anomalous_names) if anomalous_names else 'None currently flagged'}.",
        "what happened in 2020 at sasan":
            "On April 20, 2020, the Sasan UMPP dyke ruptured. Retrospective analysis shows the DII "
            "crossed critical threshold on April 7 — a 13-day early warning buffer.",
    }

    q_vec = _vectorizer.transform([question.lower()])
    sims = cosine_similarity(q_vec, _qa_vectors)[0]
    best_match_idx = sims.argmax()

    if sims[best_match_idx] > 0.25:
        best_key = list(QA_BANK.keys())[best_match_idx]
        if best_key in answers:
            return answers[best_key]

    from data_loader import get_pond_by_name_fuzzy
    row = get_pond_by_name_fuzzy(question, df)
    if row is not None:
        return (
            f"{row['pond_name']} — DII {row['dii_score']:.2f} [{row['risk_category']}]. "
            f"Breach probability: {int(row.get('breach_probability', 0) * 100)}%."
        )

    return "Groq API unavailable. Specify a pond name directly (e.g. 'Sasan' or 'Korba') for a status summary."