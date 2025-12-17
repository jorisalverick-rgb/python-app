# ai.py
"""
Python Quest — IA layer (Gemini principal + Groq secours) + JSON validation

Objectif: "ça marche une bonne fois pour toute".
- Gemini par défaut (meilleur)
- Fallback Groq automatique si Gemini est saturé / quota / indisponible
- Sélection manuelle possible: auto | gemini | groq
- Compat .env: GROQ_* ou LLM_* (comme ton autre app R)
- Extraction JSON robuste + réparation JSON (2 passes) quel que soit le provider
- Tout le texte joueur en français
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import requests
from dotenv import load_dotenv

load_dotenv()  # Streamlit does not auto-load .env


# =========================
# Configuration
# =========================

# --- Gemini ---
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
GEMINI_API_VERSION = os.getenv("GEMINI_API_VERSION", "v1beta").strip()
GEMINI_BASE = f"https://generativelanguage.googleapis.com/{GEMINI_API_VERSION}"
GEMINI_MODELS_URL = f"{GEMINI_BASE}/models"
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "").strip()

# --- Groq (OpenAI-compatible) ---
# On accepte:
#   - GROQ_API_KEY / GROQ_BASE_URL / GROQ_MODEL
#   - OU LLM_API_KEY / LLM_BASE_URL / LLM_MODEL (ton format existant)
GROQ_API_KEY_ENV = os.getenv("GROQ_API_KEY_ENV", "GROQ_API_KEY").strip()

GROQ_BASE_URL = (
    os.getenv("GROQ_BASE_URL")
    or os.getenv("LLM_BASE_URL")
    or "https://api.groq.com/openai/v1"
).strip().rstrip("/")

GROQ_MODEL = (
    os.getenv("GROQ_MODEL")
    or os.getenv("LLM_MODEL")
    or "llama-3.1-8b-instant"
).strip()

# Networking
HTTP_TIMEOUT = int(os.getenv("GEMINI_HTTP_TIMEOUT", "90"))
MAX_HTTP_ATTEMPTS = int(os.getenv("GEMINI_HTTP_RETRIES", "2"))

# Provider choices
PROVIDERS = ("auto", "gemini", "groq")

# JSON parsing helpers
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_FENCE_START_RE = re.compile(r"^```[a-zA-Z]*\s*")
_FENCE_END_RE = re.compile(r"\s*```$")


# =========================
# Errors
# =========================

class AIError(Exception):
    pass

class AIResponseFormatError(AIError):
    pass

class AIRequestError(AIError):
    pass


# =========================
# JSON utilities
# =========================

def _strip_code_fences(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = _FENCE_START_RE.sub("", t)
        t = _FENCE_END_RE.sub("", t).strip()
    return t.strip()

def _extract_json_object(text: str) -> Dict[str, Any]:
    """
    Robust JSON extraction:
    - direct dict JSON
    - JSON list -> first dict
    - fenced ```json ... ```
    - first {...} block inside any surrounding text
    """
    raw = _strip_code_fences((text or "").strip())

    # 1) Direct parse
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, list) and obj and isinstance(obj[0], dict):
            return obj[0]
    except Exception:
        pass

    # 2) Find first {...} block
    m = _JSON_OBJECT_RE.search(raw)
    if not m:
        raise AIResponseFormatError("No JSON object found in model output.")
    block = m.group(0)

    try:
        obj = json.loads(block)
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, list) and obj and isinstance(obj[0], dict):
            return obj[0]
    except Exception as e:
        raise AIResponseFormatError(f"Failed to parse extracted JSON: {e}") from e

    raise AIResponseFormatError("Extracted JSON is not a dict.")

def _require_keys(obj: Dict[str, Any], keys: List[str], ctx: str = "") -> None:
    missing = [k for k in keys if k not in obj]
    if missing:
        raise AIResponseFormatError(f"Missing keys {missing} in JSON response. Context={ctx}")

def _clamp_int(x: Any, lo: int, hi: int, default: int) -> int:
    try:
        v = int(x)
    except Exception:
        return default
    return max(lo, min(hi, v))


# =========================
# Player profile (compact)
# =========================

@dataclass
class PlayerProfile:
    player_name: str
    level: int
    difficulty: int
    zone: str
    recent_summary: str
    weak_topics: List[str]
    preferred_challenge_types: List[str]

def build_profile_payload(game_state: Dict[str, Any]) -> PlayerProfile:
    name = str(game_state.get("player_name", "Joris"))
    level = int(game_state.get("level", 1))
    difficulty = int(game_state.get("difficulty", 4))
    zone = str(game_state.get("zone", "Village des Bases"))

    hist = game_state.get("history", []) or []
    recent = hist[-6:]
    if recent:
        parts = [
            f"{s.get('topic','mixed')}|{s.get('challenge_type','mixed')}:{s.get('verdict','?')}"
            for s in recent
        ]
        recent_summary = " ; ".join(parts)
    else:
        recent_summary = "Aucune tentative pour l’instant."

    stats = game_state.get("stats", {}) or {}
    topic_attempts = stats.get("topic_attempts", {}) or {}
    topic_wrongs = stats.get("topic_wrongs", {}) or {}

    weak = []
    for t, a in topic_attempts.items():
        a = int(a or 0)
        w = int(topic_wrongs.get(t, 0) or 0)
        if a >= 3:
            rate = (w + 1) / (a + 3)
            if rate >= 0.45:
                weak.append((rate, t))
    weak.sort(reverse=True)
    weak_topics = [t for _, t in weak[:3]] or []

    return PlayerProfile(
        player_name=name,
        level=level,
        difficulty=difficulty,
        zone=zone,
        recent_summary=recent_summary,
        weak_topics=weak_topics,
        preferred_challenge_types=["code_builder", "debug_arena"],
    )


# =========================
# Prompts (FR)
# =========================

def system_rules_text() -> str:
    return (
        "Tu es le Game Master + Judge + Coach du jeu 'Python Quest'.\n"
        "Langue OBLIGATOIRE: français (FR). Le joueur répond en français.\n"
        "Ne réponds pas en anglais (sauf mots-clés Python dans du code).\n"
        "Ton: ludique, enfant-friendly, motivant, mais sérieux intellectuellement.\n"
        "Ne jamais utiliser les mots: devoir, examen, interro, TD, TP.\n"
        "Priorité: défis où le joueur ÉCRIT du code, puis débogage et mini quiz.\n"
        "Sortie OBLIGATOIRE: un SEUL objet JSON valide.\n"
        "Aucun markdown. Aucun texte hors JSON.\n"
        "La sortie doit commencer par '{' et finir par '}'.\n"
    )

def build_scene_prompt(profile: PlayerProfile, next_hint: Dict[str, Any]) -> str:
    topic = str(next_hint.get("topic", "mixed"))
    ctype = str(next_hint.get("challenge_type", "code_builder"))
    difficulty = int(next_hint.get("difficulty", profile.difficulty))
    weak = ", ".join(profile.weak_topics) if profile.weak_topics else "aucun"

    schema = (
        "{\n"
        '  "scene_id": "string",\n'
        '  "zone": "string",\n'
        '  "topic": "basics|data_structures|control_flow|functions|oop|pandas|matplotlib|mixed",\n'
        '  "challenge_type": "code_builder|debug_arena|qcm|true_false|boss_fight",\n'
        '  "difficulty": 1,\n'
        '  "narration": "string (FR)",\n'
        '  "mission_title": "string (FR, court)",\n'
        '  "mission_goal": "string (FR, 1-2 lignes)",\n'
        '  "input_mode": "code|text|choice|true_false",\n'
        '  "prompt": "string (FR, consigne claire)",\n'
        '  "starter_code": "string (optionnel, Python)",\n'
        '  "choices": ["A ...","B ...","C ...","D ..."] (optionnel),\n'
        '  "constraints": ["..."] (FR),\n'
        '  "tests": [{"input":"string","expected_contains":"string"}],\n'
        '  "rubric": {"primary_skills":["..."],"common_mistakes":["..."],"grading_focus":"string (FR)"}\n'
        "}\n"
    )

    return (
        f"{system_rules_text()}\n"
        "TÂCHE: Génère la prochaine scène jouable + un défi.\n"
        "Retourne UNIQUEMENT le JSON du schéma.\n\n"
        "PROFIL JOUEUR:\n"
        f"- nom: {profile.player_name}\n"
        f"- niveau: {profile.level}\n"
        f"- difficulté_globale: {profile.difficulty}\n"
        f"- zone: {profile.zone}\n"
        f"- tentatives_récentes: {profile.recent_summary}\n"
        f"- points_faibles: {weak}\n\n"
        "INDICATION SUIVANTE:\n"
        f"- topic: {topic}\n"
        f"- challenge_type: {ctype}\n"
        f"- difficulty: {difficulty}\n\n"
        f"SCHÉMA:\n{schema}\n"
        "RÈGLE ABSOLUE: commence par '{' et finis par '}'. Un seul objet JSON.\n"
    )

def build_judge_prompt(profile: PlayerProfile, scene: Dict[str, Any], player_answer: str) -> str:
    compact_scene = {
        "scene_id": scene.get("scene_id"),
        "topic": scene.get("topic"),
        "challenge_type": scene.get("challenge_type"),
        "difficulty": scene.get("difficulty"),
        "input_mode": scene.get("input_mode"),
        "mission_title": scene.get("mission_title"),
        "mission_goal": scene.get("mission_goal"),
        "prompt": scene.get("prompt"),
        "starter_code": scene.get("starter_code", ""),
        "choices": scene.get("choices", []),
        "constraints": scene.get("constraints", []),
        "tests": scene.get("tests", []),
        "rubric": scene.get("rubric", {}),
    }
    weak = ", ".join(profile.weak_topics) if profile.weak_topics else "aucun"

    schema = (
        "{\n"
        '  "scene_id": "string",\n'
        '  "verdict": "correct|close|wrong",\n'
        '  "xp_suggestion": 0,\n'
        '  "short_feedback": "string (FR, 1-2 lignes)",\n'
        '  "detailed_feedback": "string (FR, explication détaillée)",\n'
        '  "corrected_solution": "string (Python / réponse)",\n'
        '  "why_it_works": "string (FR)",\n'
        '  "mini_lesson": {"title":"string (FR)","content":"string (FR, très détaillé si faux)","examples":["..."]},\n'
        '  "common_pitfalls": ["..."] (FR),\n'
        '  "next_tip": "string (FR)",\n'
        '  "followup": {"suggested_next_challenge":"code_builder|debug_arena|qcm|true_false|boss_fight",'
        '              "suggested_topic":"basics|data_structures|control_flow|functions|oop|pandas|matplotlib|mixed",'
        '              "difficulty_adjustment_hint":"up|down|same"}\n'
        "}\n"
    )

    return (
        f"{system_rules_text()}\n"
        "TÂCHE: Corrige et évalue la réponse du joueur.\n"
        "Retourne UNIQUEMENT le JSON du schéma.\n\n"
        "PROFIL JOUEUR:\n"
        f"- nom: {profile.player_name}\n"
        f"- niveau: {profile.level}\n"
        f"- difficulté_globale: {profile.difficulty}\n"
        f"- points_faibles: {weak}\n\n"
        f"SCÈNE:\n{json.dumps(compact_scene, ensure_ascii=False)}\n\n"
        f"RÉPONSE DU JOUEUR:\n{player_answer}\n\n"
        f"SCHÉMA:\n{schema}\n"
        "RÈGLE ABSOLUE: commence par '{' et finis par '}'. Un seul objet JSON.\n"
    )


# =========================
# Provider selection helpers
# =========================

def _normalize_provider(value: Optional[str]) -> str:
    v = (value or "auto").lower().strip()
    return v if v in PROVIDERS else "auto"

def _should_fallback_to_groq(err: Exception) -> bool:
    """
    On bascule vers Groq quand Gemini est:
    - quota / rate limit (429, RESOURCE_EXHAUSTED)
    - surchargé (503, UNAVAILABLE, overloaded)
    - ou problème réseau / déconnexion
    """
    msg = (str(err) or "").lower()

    keywords = [
        "429", "resource_exhausted", "quota", "rate limit",
        "503", "unavailable", "overloaded",
        "remote end closed", "remote disconnected", "connection aborted",
        "timed out", "timeout", "connection reset",
    ]
    return any(k in msg for k in keywords)


# =========================
# Gemini client (auto-pick + retry)
# =========================

_cached_model: Optional[str] = None

def _get_gemini_key() -> str:
    key = os.getenv(GEMINI_API_KEY_ENV, "").strip()
    if not key:
        raise AIRequestError(f"Clé API Gemini manquante. Mets {GEMINI_API_KEY_ENV} dans ton fichier .env.")
    return key

def _list_models(timeout: int = 20) -> Dict[str, Any]:
    key = _get_gemini_key()
    url = f"{GEMINI_MODELS_URL}?key={key}"
    r = requests.get(url, timeout=timeout)
    if r.status_code >= 400:
        raise AIRequestError(f"Gemini ListModels error {r.status_code}: {r.text[:1200]}")
    return r.json()

def _pick_model_from_list(models_json: Dict[str, Any]) -> str:
    models = models_json.get("models", []) or []
    candidates: List[str] = []
    for m in models:
        name = m.get("name", "")
        methods = m.get("supportedGenerationMethods", []) or []
        if name and "generateContent" in methods:
            candidates.append(name)

    if not candidates:
        raise AIRequestError("Aucun modèle Gemini disponible ne supporte generateContent pour cette clé.")

    def score(n: str) -> int:
        n = n.lower()
        if "flash" in n:
            return 0
        if "pro" in n:
            return 1
        return 2

    candidates.sort(key=score)
    return candidates[0]

def _resolve_model(model: str) -> str:
    global _cached_model
    override = (model or "").strip()
    if override:
        if override.startswith("models/"):
            return override
        return "models/" + override

    if _cached_model:
        return _cached_model

    picked = _pick_model_from_list(_list_models())
    _cached_model = picked
    return picked

def _extract_text_from_gemini_response(data: Dict[str, Any]) -> str:
    try:
        cand0 = (data.get("candidates") or [])[0]
        content = cand0.get("content") or {}
        parts = content.get("parts") or []
        texts = []
        for p in parts:
            t = p.get("text")
            if isinstance(t, str) and t.strip():
                texts.append(t)
        return "\n".join(texts).strip()
    except Exception:
        return ""

def _gemini_generate_raw_text(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.6,
    max_output_tokens: int = 1400,
    timeout: int = HTTP_TIMEOUT,
) -> str:
    key = _get_gemini_key()
    resolved_model = _resolve_model(model)
    url = f"{GEMINI_BASE}/{resolved_model}:generateContent?key={key}"

    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": float(temperature),
            "maxOutputTokens": int(max_output_tokens),
            "responseMimeType": "application/json",
        },
    }

    last_err: Optional[Exception] = None
    for attempt in range(1, MAX_HTTP_ATTEMPTS + 1):
        try:
            r = requests.post(url, json=payload, timeout=timeout)
            if r.status_code >= 400:
                raise AIRequestError(f"Gemini API error {r.status_code}: {r.text[:2000]}")
            data = r.json()
            text = _extract_text_from_gemini_response(data)
            if not text:
                raise AIRequestError(f"Gemini returned empty text. Raw={str(data)[:1200]}")
            return text
        except requests.exceptions.RequestException as e:
            last_err = e
            time.sleep(0.8 * attempt)

    raise AIRequestError(f"Connexion instable vers Gemini (après retry): {last_err}")


# =========================
# Groq client (OpenAI-compatible)
# =========================

def _get_groq_key() -> str:
    # 1) clé via GROQ_API_KEY_ENV (par défaut "GROQ_API_KEY")
    key = os.getenv(GROQ_API_KEY_ENV, "").strip()
    if key:
        return key
    # 2) fallback vers ton format LLM_API_KEY
    key2 = os.getenv("LLM_API_KEY", "").strip()
    if key2:
        return key2
    raise AIRequestError(
        f"Clé API Groq manquante. Mets {GROQ_API_KEY_ENV}=... (ou LLM_API_KEY=...) dans ton fichier .env."
    )

def _groq_generate_raw_text(
    prompt: str,
    model: str = GROQ_MODEL,
    temperature: float = 0.4,
    max_output_tokens: int = 1400,
    timeout: int = HTTP_TIMEOUT,
) -> str:
    """
    Groq = endpoint OpenAI-compatible: POST /chat/completions
    On garde la consigne JSON dans le prompt + notre repair derrière.
    """
    key = _get_groq_key()
    url = f"{GROQ_BASE_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": float(temperature),
        "max_tokens": int(max_output_tokens),
    }

    last_err: Optional[Exception] = None
    for attempt in range(1, MAX_HTTP_ATTEMPTS + 1):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if r.status_code >= 400:
                raise AIRequestError(f"Groq API error {r.status_code}: {r.text[:2000]}")
            data = r.json()
            try:
                text = data["choices"][0]["message"]["content"]
            except Exception as e:
                raise AIRequestError(f"Format Groq inattendu: {e}. Raw={str(data)[:1200]}")
            if not isinstance(text, str) or not text.strip():
                raise AIRequestError("Groq returned empty text.")
            return text.strip()
        except requests.exceptions.RequestException as e:
            last_err = e
            time.sleep(0.8 * attempt)

    raise AIRequestError(f"Connexion instable vers Groq (après retry): {last_err}")


# =========================
# Auto-repair JSON (2 passes)
# =========================

def _repair_prompt(schema_name: str, schema_text: str, raw_text: str, strict_level: int) -> str:
    if strict_level == 2:
        extra = (
            "IMPORTANT:\n"
            "- Tu dois produire EXACTEMENT 1 objet JSON.\n"
            "- Aucun texte avant/après.\n"
            "- Aucune liste JSON.\n"
            "- Commence par '{' et finis par '}'.\n"
            "- Toutes les chaînes doivent être en FR.\n"
        )
    else:
        extra = "Rappel: un SEUL objet JSON, commence par '{' et finis par '}'.\n"

    return (
        f"{system_rules_text()}\n"
        f"TÂCHE: Transforme le contenu ci-dessous en UN SEUL objet JSON valide.\n"
        f"Nom du schéma: {schema_name}\n"
        f"{extra}\n"
        f"SCHÉMA CIBLE:\n{schema_text}\n\n"
        f"CONTENU À CONVERTIR:\n{raw_text}\n"
    )

def _repair_to_json(
    raw_text: str,
    schema_name: str,
    schema_text: str,
    use_groq: bool = False,
) -> Dict[str, Any]:
    """
    Repair via the same provider we used:
    - if use_groq=True => repair with Groq (avoid hitting Gemini quota again)
    - else => repair with Gemini
    """
    gen = _groq_generate_raw_text if use_groq else _gemini_generate_raw_text

    p1 = _repair_prompt(schema_name, schema_text, raw_text, strict_level=1)
    t1 = gen(prompt=p1, temperature=0.2, max_output_tokens=1600)
    try:
        return _extract_json_object(t1)
    except AIResponseFormatError:
        p2 = _repair_prompt(schema_name, schema_text, raw_text, strict_level=2)
        t2 = gen(prompt=p2, temperature=0.0, max_output_tokens=1600)
        return _extract_json_object(t2)


# =========================
# Core generator: manual selection + fallback
# =========================

def _generate_raw_text_with_provider_choice(
    prompt: str,
    temperature: float,
    max_tokens: int,
    preferred_provider: str = "auto",
) -> Tuple[str, str]:
    """
    Returns (raw_text, provider_used) where provider_used is "gemini" or "groq".

    preferred_provider:
      - "gemini": Gemini only (no fallback)
      - "groq":   Groq only (no fallback)
      - "auto":   Gemini then fallback Groq if Gemini fails with known transient/quota errors
    """
    preferred_provider = _normalize_provider(preferred_provider)

    def call_gemini() -> Tuple[str, str]:
        return (
            _gemini_generate_raw_text(prompt=prompt, temperature=temperature, max_output_tokens=max_tokens),
            "gemini",
        )

    def call_groq() -> Tuple[str, str]:
        return (
            _groq_generate_raw_text(prompt=prompt, temperature=temperature, max_output_tokens=max_tokens),
            "groq",
        )

    if preferred_provider == "gemini":
        return call_gemini()

    if preferred_provider == "groq":
        return call_groq()

    # auto
    try:
        return call_gemini()
    except Exception as e:
        # Si Gemini échoue (quota, overload, réseau), on tente Groq
        if _should_fallback_to_groq(e):
            return call_groq()
        raise


def _generate_json_with_repair(
    prompt: str,
    required_keys: List[str],
    ctx: str,
    schema_text: str,
    temperature: float,
    max_tokens: int,
    preferred_provider: str = "auto",
) -> Dict[str, Any]:
    raw, provider = _generate_raw_text_with_provider_choice(
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        preferred_provider=preferred_provider,
    )

    try:
        obj = _extract_json_object(raw)
    except AIResponseFormatError:
        obj = _repair_to_json(raw, schema_name=ctx, schema_text=schema_text, use_groq=(provider == "groq"))

    _require_keys(obj, required_keys, ctx=ctx)
    return obj


# =========================
# Public API
# =========================

def generate_scene(
    game_state: Dict[str, Any],
    next_hint: Dict[str, Any],
    preferred_provider: Optional[str] = None,
) -> Dict[str, Any]:
    """
    preferred_provider:
      - None => utilise game_state["ai_provider"] si présent, sinon "auto"
      - "auto" | "gemini" | "groq"
    """
    if preferred_provider is None:
        preferred_provider = game_state.get("ai_provider", "auto")

    profile = build_profile_payload(game_state)
    prompt = build_scene_prompt(profile, next_hint)

    schema_text = (
        "{\n"
        '  "scene_id": "string",\n'
        '  "zone": "string",\n'
        '  "topic": "basics|data_structures|control_flow|functions|oop|pandas|matplotlib|mixed",\n'
        '  "challenge_type": "code_builder|debug_arena|qcm|true_false|boss_fight",\n'
        '  "difficulty": 1,\n'
        '  "narration": "string",\n'
        '  "mission_title": "string",\n'
        '  "mission_goal": "string",\n'
        '  "input_mode": "code|text|choice|true_false",\n'
        '  "prompt": "string",\n'
        '  "starter_code": "string (optional)",\n'
        '  "choices": ["A","B","C","D"] (optional),\n'
        '  "constraints": ["..."],\n'
        '  "tests": [{"input":"string","expected_contains":"string"}],\n'
        '  "rubric": {"primary_skills":["..."],"common_mistakes":["..."],"grading_focus":"string"}\n'
        "}\n"
    )

    required = [
        "scene_id", "zone", "topic", "challenge_type", "difficulty", "narration",
        "mission_title", "mission_goal", "input_mode", "prompt", "constraints", "rubric"
    ]

    obj = _generate_json_with_repair(
        prompt=prompt,
        required_keys=required,
        ctx="SCENE",
        schema_text=schema_text,
        temperature=0.6,
        max_tokens=1500,
        preferred_provider=str(preferred_provider),
    )

    obj["difficulty"] = _clamp_int(
        obj.get("difficulty"), 1, 10,
        default=int(next_hint.get("difficulty", profile.difficulty))
    )
    obj.setdefault("starter_code", "")
    obj.setdefault("choices", [])
    obj.setdefault("tests", [])
    return obj

def judge_answer(
    game_state: Dict[str, Any],
    scene: Dict[str, Any],
    player_answer: str,
    preferred_provider: Optional[str] = None,
) -> Dict[str, Any]:
    """
    preferred_provider:
      - None => utilise game_state["ai_provider"] si présent, sinon "auto"
      - "auto" | "gemini" | "groq"
    """
    if preferred_provider is None:
        preferred_provider = game_state.get("ai_provider", "auto")

    profile = build_profile_payload(game_state)
    prompt = build_judge_prompt(profile, scene, player_answer)

    schema_text = (
        "{\n"
        '  "scene_id": "string",\n'
        '  "verdict": "correct|close|wrong",\n'
        '  "xp_suggestion": 0,\n'
        '  "short_feedback": "string",\n'
        '  "detailed_feedback": "string",\n'
        '  "corrected_solution": "string",\n'
        '  "why_it_works": "string",\n'
        '  "mini_lesson": {"title":"string","content":"string","examples":["..."]},\n'
        '  "common_pitfalls": ["..."],\n'
        '  "next_tip": "string",\n'
        '  "followup": {"suggested_next_challenge":"code_builder|debug_arena|qcm|true_false|boss_fight",'
        '              "suggested_topic":"basics|data_structures|control_flow|functions|oop|pandas|matplotlib|mixed",'
        '              "difficulty_adjustment_hint":"up|down|same"}\n'
        "}\n"
    )

    required = [
        "scene_id", "verdict", "xp_suggestion", "short_feedback", "detailed_feedback",
        "corrected_solution", "why_it_works", "mini_lesson", "common_pitfalls", "next_tip", "followup"
    ]

    obj = _generate_json_with_repair(
        prompt=prompt,
        required_keys=required,
        ctx="JUDGE",
        schema_text=schema_text,
        temperature=0.25,
        max_tokens=1800,
        preferred_provider=str(preferred_provider),
    )

    verdict = str(obj.get("verdict", "wrong")).lower().strip()
    if verdict not in ("correct", "close", "wrong"):
        verdict = "wrong"
    obj["verdict"] = verdict
    obj["xp_suggestion"] = _clamp_int(obj.get("xp_suggestion"), 0, 50, default=0)

    ml = obj.get("mini_lesson", {})
    if not isinstance(ml, dict):
        ml = {"title": "Mini-cours", "content": "", "examples": []}
    ml.setdefault("title", "Mini-cours")
    ml.setdefault("content", "")
    ml.setdefault("examples", [])
    if not isinstance(ml.get("examples"), list):
        ml["examples"] = [str(ml.get("examples"))]
    obj["mini_lesson"] = ml

    fu = obj.get("followup", {})
    if not isinstance(fu, dict):
        fu = {"suggested_next_challenge": "code_builder", "suggested_topic": "mixed", "difficulty_adjustment_hint": "same"}
    fu.setdefault("suggested_next_challenge", "code_builder")
    fu.setdefault("suggested_topic", "mixed")
    fu.setdefault("difficulty_adjustment_hint", "same")
    obj["followup"] = fu

    return obj


def quick_heuristic_checks(scene: Dict[str, Any], player_answer: str) -> Tuple[bool, List[str]]:
    msgs: List[str] = []
    ans = (player_answer or "").strip()
    if not ans:
        return False, ["Tu n'as rien envoyé 😅 Écris au moins un début et je t'aide."]

    mode = str(scene.get("input_mode", "code"))
    if mode == "code":
        p = str(scene.get("prompt", "")).lower()
        if "classe" in p and "class " not in ans:
            msgs.append("Indice: tu vas sûrement créer une classe (`class ...:`).")
        if "fonction" in p and "def " not in ans:
            msgs.append("Indice: tu vas sûrement définir une fonction (`def ...():`).")
    return True, msgs
