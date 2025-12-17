# app.py
"""
Python Quest — Streamlit app (UI + game loop)

Run:
  streamlit run app.py

Files required:
- game.py
- ai.py
- storage.py
- .env (contains GEMINI_API_KEY and/or Groq keys)
"""

from __future__ import annotations

import os
import traceback
from typing import Any, Dict, Optional, Tuple, List

import streamlit as st

from game import PlayerState, recommended_next, apply_judgement, is_game_over, xp_progress_within_level
from storage import load_state_dict, save_state_dict, reset_save
from ai import generate_scene, judge_answer, quick_heuristic_checks, AIError

import os
import streamlit as st

st.sidebar.write("DEBUG: GEMINI_API_KEY existe ?", bool(os.getenv("GEMINI_API_KEY")))
st.sidebar.write("DEBUG: LLM_API_KEY existe ?", bool(os.getenv("LLM_API_KEY")))

st.sidebar.write("DEBUG GEMINI_API_KEY:", bool(os.getenv("GEMINI_API_KEY")))
st.sidebar.write("DEBUG LLM_API_KEY:", bool(os.getenv("LLM_API_KEY")))

# -----------------------------
# App config
# -----------------------------

APP_TITLE = "Python Quest"
SAVE_PATH = "save.json"

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------
# Styling (kid-friendly)
# -----------------------------

def inject_css() -> None:
    st.markdown(
        """
<style>
/* Global */
:root { --pq-radius: 22px; --pq-pad: 16px; }

.block-container { padding-top: 1.2rem; padding-bottom: 1.5rem; }
h1, h2, h3 { letter-spacing: 0.2px; }

/* Cards */
.pq-card{
  border-radius: var(--pq-radius);
  padding: var(--pq-pad);
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.05);
  box-shadow: 0 10px 22px rgba(0,0,0,0.18);
}
.pq-title{
  font-size: 1.15rem;
  font-weight: 700;
  margin-bottom: 0.35rem;
}
.pq-sub{
  opacity: 0.85;
  margin-bottom: 0.65rem;
}
.pq-pill{
  display: inline-block;
  padding: 0.15rem 0.6rem;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.18);
  background: rgba(255,255,255,0.06);
  font-size: 0.82rem;
  margin-right: 0.35rem;
  margin-bottom: 0.35rem;
}

/* Mission prompt box */
.pq-prompt{
  border-radius: var(--pq-radius);
  padding: 16px;
  border: 1px dashed rgba(255,255,255,0.22);
  background: rgba(0,0,0,0.15);
}

/* Buttons */
.stButton > button{
  border-radius: 14px;
  padding: 0.65rem 0.9rem;
  font-weight: 650;
}
.stTextArea textarea{
  border-radius: 14px;
}

/* Code blocks */
code, pre { border-radius: 14px !important; }

/* Small footer */
.pq-footer{
  opacity: 0.65;
  font-size: 0.85rem;
}
</style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# Session state helpers
# -----------------------------

def ss_init_defaults() -> None:
    if "state" not in st.session_state:
        st.session_state.state = None  # PlayerState
    if "scene" not in st.session_state:
        st.session_state.scene = None  # dict from AI
    if "answer" not in st.session_state:
        st.session_state.answer = ""
    if "last_judge" not in st.session_state:
        st.session_state.last_judge = None  # dict
    if "last_engine_update" not in st.session_state:
        st.session_state.last_engine_update = None  # dict
    if "busy" not in st.session_state:
        st.session_state.busy = False

    # NEW: IA provider selection
    if "ai_provider" not in st.session_state:
        st.session_state.ai_provider = "auto"  # auto | gemini | groq


def load_or_create_player() -> PlayerState:
    saved = load_state_dict(SAVE_PATH)
    if saved:
        try:
            state = PlayerState.from_dict(saved)
            return state
        except Exception:
            # corrupted save or incompatible
            pass
    return PlayerState(player_name="Joris")


def persist_player(state: PlayerState) -> None:
    try:
        save_state_dict(state.to_dict(), SAVE_PATH)
    except Exception as e:
        st.warning(f"⚠️ Sauvegarde impossible: {e}")


# -----------------------------
# UI widgets
# -----------------------------

def hearts_row(hearts: int, max_hearts: int) -> str:
    full = "❤️" * max(0, min(hearts, max_hearts))
    empty = "🖤" * max(0, max_hearts - max(0, min(hearts, max_hearts)))
    return full + empty


def render_header(state: PlayerState) -> None:
    lvl, xp_into, pct = xp_progress_within_level(state.xp)
    st.title("🧩 Python Quest")
    st.caption("Apprendre Python en jouant — aventure + code + boss fights ✨")

    col1, col2, col3, col4 = st.columns([1.2, 1.0, 1.2, 1.2], vertical_alignment="center")
    with col1:
        st.markdown(
            f"""
<div class="pq-card">
  <div class="pq-title">👤 {state.player_name}</div>
  <div class="pq-sub">Zone : <b>{state.zone}</b></div>
  <div class="pq-pill">Niveau {state.level}</div>
  <div class="pq-pill">Difficulté {state.difficulty}/10</div>
</div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
<div class="pq-card">
  <div class="pq-title">❤️ Vies</div>
  <div style="font-size:1.25rem">{hearts_row(state.hearts, state.max_hearts)}</div>
  <div class="pq-sub">Combo : <b>{state.streak}</b> (Best: {state.best_streak})</div>
</div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
<div class="pq-card">
  <div class="pq-title">⚡ XP</div>
  <div class="pq-sub">{state.xp} XP total</div>
</div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(pct, text=f"Progression niveau {lvl} → {lvl+1} : {int(pct*100)}% ({xp_into} XP dans le niveau)")
    with col4:
        inv = state.inventory[-6:] if state.inventory else []
        badges = state.badges[-4:] if state.badges else []
        st.markdown('<div class="pq-card">', unsafe_allow_html=True)
        st.markdown('<div class="pq-title">🎒 Inventaire & Badges</div>', unsafe_allow_html=True)
        if inv:
            st.markdown(" ".join([f"<span class='pq-pill'>{x}</span>" for x in inv]), unsafe_allow_html=True)
        else:
            st.caption("Inventaire vide… pour l’instant 😄")
        if badges:
            st.markdown(" ".join([f"<span class='pq-pill'>🏅 {b}</span>" for b in badges]), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


def _provider_label(p: str) -> str:
    return {"auto": "Auto (Gemini → Groq)", "gemini": "Gemini (forcé)", "groq": "Groq (forcé)"}.get(p, "Auto")


def render_sidebar(state: PlayerState) -> None:
    with st.sidebar:
        st.header("⚙️ Commandes")
        st.write("Ajuste seulement si tu veux. Sinon, laisse le jeu gérer.")

        # -------- NEW: IA provider control --------
        st.subheader("🤖 Moteur IA")
        st.session_state.ai_provider = st.radio(
            "Choix IA",
            options=["auto", "gemini", "groq"],
            format_func=_provider_label,
            index=["auto", "gemini", "groq"].index(st.session_state.ai_provider),
            horizontal=False,
        )

        cprov1, cprov2 = st.columns(2)
        with cprov1:
            if st.button("🔁 Toggle"):
                # Toggle rapide gemini <-> groq (auto reste au choix via radio)
                if st.session_state.ai_provider == "gemini":
                    st.session_state.ai_provider = "groq"
                elif st.session_state.ai_provider == "groq":
                    st.session_state.ai_provider = "gemini"
                else:
                    # si auto, on passe à gemini pour commencer
                    st.session_state.ai_provider = "gemini"
                st.rerun()

        with cprov2:
            st.caption(f"Actuel: **{_provider_label(st.session_state.ai_provider)}**")

        st.divider()

        # Manual difficulty override (optional)
        new_diff = st.slider("Difficulté globale", 1, 10, int(state.difficulty))
        if new_diff != state.difficulty:
            state.difficulty = int(new_diff)
            persist_player(state)
            st.success("Difficulté mise à jour ✅")

        st.divider()
        st.subheader("🧼 Sauvegarde")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("💾 Forcer sauvegarde"):
                persist_player(state)
                st.success("Sauvegardé.")
        with c2:
            if st.button("🗑️ Reset jeu"):
                reset_save(SAVE_PATH)
                st.session_state.state = PlayerState(player_name=state.player_name)
                st.session_state.scene = None
                st.session_state.answer = ""
                st.session_state.last_judge = None
                st.session_state.last_engine_update = None
                st.success("Jeu réinitialisé ✅")
                st.rerun()

        st.divider()
        st.subheader("🧠 À propos")
        st.caption("Python Quest génère des missions via IA (Gemini/Groq), puis corrige avec un feedback détaillé.")
        if not os.getenv("GEMINI_API_KEY", "").strip():
            st.warning("Clé GEMINI_API_KEY manquante dans .env / environnement.")
        # Groq est optionnel en mode auto, mais utile
        if not (os.getenv("GROQ_API_KEY", "").strip() or os.getenv("LLM_API_KEY", "").strip()):
            st.info("Astuce: ajoute une clé Groq (GROQ_API_KEY ou LLM_API_KEY) pour éviter les blocages quota Gemini.")


def render_scene(scene: Dict[str, Any], state: PlayerState) -> None:
    # Mission card
    tags = [
        f"📌 {scene.get('topic','mixed')}",
        f"🎮 {scene.get('challenge_type','code_builder')}",
        f"🔥 diff {scene.get('difficulty', state.difficulty)}/10",
    ]
    tags_html = " ".join([f"<span class='pq-pill'>{t}</span>" for t in tags])

    st.markdown(
        f"""
<div class="pq-card">
  <div class="pq-title">🗺️ {scene.get('mission_title','Mission')}</div>
  <div class="pq-sub">{scene.get('narration','')}</div>
  {tags_html}
  <div class="pq-prompt" style="margin-top:12px">
    <b>🎯 Objectif</b><br/>
    {scene.get('mission_goal','')}
    <hr style="opacity:0.2"/>
    <b>🧩 Ta mission</b><br/>
    {scene.get('prompt','')}
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    # Starter code
    starter = scene.get("starter_code", "") or ""
    if starter.strip():
        st.markdown("#### 🧱 Starter code")
        st.code(starter, language="python")

    # Constraints
    constraints = scene.get("constraints", []) or []
    if constraints:
        st.markdown("#### ✅ Contraintes")
        st.write(" • " + "\n • ".join([str(c) for c in constraints]))


def render_answer_input(scene: Dict[str, Any]) -> str:
    mode = str(scene.get("input_mode", "code"))

    if mode in ("choice", "true_false"):
        # Present choices as radio
        if mode == "true_false":
            choice = st.radio("Ta réponse :", ["Vrai", "Faux"], horizontal=True)
            return "true" if choice == "Vrai" else "false"

        choices = scene.get("choices", []) or []
        if not choices:
            st.info("Aucun choix fourni (mission mal formée). Clique 'Nouvelle mission'.")
            return ""
        picked = st.radio("Choisis une option :", choices)
        return picked

    # code/text input
    label = "✍️ Écris ton code ici" if mode == "code" else "✍️ Ta réponse"
    height = 260 if mode == "code" else 120
    return st.text_area(label, value=st.session_state.answer, height=height, placeholder="Écris ici…")


def render_feedback(judge: Dict[str, Any], engine_update: Dict[str, Any], state: PlayerState) -> None:
    verdict = judge.get("verdict", "wrong")
    short = judge.get("short_feedback", "")
    xp_sug = judge.get("xp_suggestion", 0)

    # Verdict banner
    if verdict == "correct":
        st.success(f"✅ {short}")
        st.balloons()
    elif verdict == "close":
        st.warning(f"🟡 {short}")
    else:
        st.error(f"❌ {short}")

    # Engine outcome
    xp_delta = engine_update.get("xp_delta", 0)
    hearts_delta = engine_update.get("hearts_delta", 0)
    level_up = engine_update.get("level_up", False)
    gained_heart = engine_update.get("gained_heart", False)
    new_badges = engine_update.get("new_badges", []) or []

    st.markdown(
        f"""
<div class="pq-card">
  <div class="pq-title">📈 Résultat</div>
  <span class="pq-pill">XP +{xp_delta}</span>
  <span class="pq-pill">Vies {hearts_delta:+d}</span>
  <span class="pq-pill">Suggestion IA: {int(xp_sug)} XP</span>
  <span class="pq-pill">Difficulté actuelle: {state.difficulty}/10</span>
</div>
        """,
        unsafe_allow_html=True,
    )

    if level_up:
        st.success(f"🎉 Niveau UP ! Tu es maintenant niveau {state.level} !")
    if gained_heart:
        st.info("❤️ Bonus : tu récupères une vie !")
    if new_badges:
        st.info("🏅 Nouveaux badges : " + ", ".join(new_badges))

    # Detailed feedback sections
    with st.expander("🧠 Explication détaillée (lisible)", expanded=True):
        st.write(judge.get("detailed_feedback", ""))

    with st.expander("✅ Solution corrigée (référence)", expanded=False):
        sol = judge.get("corrected_solution", "")
        if sol.strip():
            st.code(sol, language="python")
        else:
            st.write(sol)

    with st.expander("🔍 Pourquoi ça marche (logique)", expanded=False):
        st.write(judge.get("why_it_works", ""))

    ml = judge.get("mini_lesson", {}) or {}
    with st.expander("📚 Mini-cours (si tu veux tout comprendre)", expanded=(verdict != "correct")):
        st.subheader(ml.get("title", "Mini-cours"))
        st.write(ml.get("content", ""))
        examples = ml.get("examples", []) or []
        if examples:
            st.markdown("**Exemples :**")
            for ex in examples[:4]:
                st.code(str(ex), language="python")

    pitfalls = judge.get("common_pitfalls", []) or []
    if pitfalls:
        with st.expander("⚠️ Pièges fréquents", expanded=False):
            st.write(" • " + "\n • ".join([str(p) for p in pitfalls]))

    st.caption("👉 Prochain tip : " + str(judge.get("next_tip", "")))


# -----------------------------
# Main flow
# -----------------------------

def ensure_scene(state: PlayerState) -> None:
    """
    If no scene exists, generate one.
    """
    if st.session_state.scene is not None:
        return

    next_hint = recommended_next(state)
    provider = st.session_state.ai_provider

    try:
        st.session_state.scene = generate_scene(state.to_dict(), next_hint, preferred_provider=provider)
        st.session_state.last_judge = None
        st.session_state.last_engine_update = None
        st.session_state.answer = ""
        persist_player(state)
    except AIError as e:
        st.session_state.scene = None
        st.error(f"🤖 Erreur IA ({_provider_label(provider)}) : {e}")
    except Exception as e:
        st.session_state.scene = None
        st.error(f"Erreur inattendue : {e}")
        st.code(traceback.format_exc())


def request_new_scene(state: PlayerState) -> None:
    st.session_state.scene = None
    ensure_scene(state)


def submit_answer(state: PlayerState) -> None:
    scene = st.session_state.scene
    if not scene:
        st.warning("Aucune mission active. Clique 'Nouvelle mission'.")
        return

    answer = st.session_state.answer or ""
    ok, msgs = quick_heuristic_checks(scene, answer)
    if not ok:
        for m in msgs:
            st.warning(m)
        return
    for m in msgs:
        st.info(m)

    # time measurement
    seconds_spent = state.stop_attempt_timer()
    provider = st.session_state.ai_provider

    try:
        judge = judge_answer(state.to_dict(), scene, answer, preferred_provider=provider)
    except AIError as e:
        st.error(f"🤖 Erreur IA ({_provider_label(provider)}) : {e}")
        return
    except Exception as e:
        st.error(f"Erreur inattendue : {e}")
        st.code(traceback.format_exc())
        return

    # Update engine
    engine_update = apply_judgement(
        state=state,
        verdict=str(judge.get("verdict", "wrong")),
        challenge_type=str(scene.get("challenge_type", "code_builder")),
        topic=str(scene.get("topic", "mixed")),
        difficulty_used=int(scene.get("difficulty", state.difficulty)),
        seconds_spent=seconds_spent,
    )

    # Persist
    persist_player(state)

    # Store results
    st.session_state.last_judge = judge
    st.session_state.last_engine_update = engine_update

    # Game over?
    if is_game_over(state):
        st.session_state.scene = None


# -----------------------------
# App
# -----------------------------

def main() -> None:
    inject_css()
    ss_init_defaults()

    # Load/create player into session
    if st.session_state.state is None:
        st.session_state.state = load_or_create_player()

    state: PlayerState = st.session_state.state
    render_sidebar(state)
    render_header(state)

    # Game Over screen
    if is_game_over(state):
        st.markdown(
            """
<div class="pq-card">
  <div class="pq-title">💀 Oh non… plus de vies !</div>
  <div class="pq-sub">Pas grave : tu vas revenir plus fort 💪</div>
</div>
            """,
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🔁 Rejouer (3 vies)"):
                state.hearts = state.max_hearts
                persist_player(state)
                st.session_state.scene = None
                st.session_state.last_judge = None
                st.session_state.last_engine_update = None
                st.rerun()
        with c2:
            if st.button("🗑️ Reset total"):
                reset_save(SAVE_PATH)
                st.session_state.state = PlayerState(player_name=state.player_name)
                st.session_state.scene = None
                st.session_state.last_judge = None
                st.session_state.last_engine_update = None
                st.rerun()
        st.stop()

    # Controls
    st.divider()
    top_c1, top_c2, top_c3 = st.columns([1, 1, 2], vertical_alignment="center")

    with top_c1:
        if st.button("🆕 Nouvelle mission"):
            request_new_scene(state)
            st.rerun()

    with top_c2:
        if st.button("🚀 Booster (plus dur)"):
            state.difficulty = min(10, state.difficulty + 1)
            persist_player(state)
            st.success("Difficulté augmentée ✅")
            request_new_scene(state)
            st.rerun()

    with top_c3:
        st.caption("Astuce: si c'est trop facile, clique Booster. Si c'est dur, la difficulté s'ajuste toute seule.")

    # Ensure current scene exists
    ensure_scene(state)
    scene = st.session_state.scene

    if not scene:
        st.info("Clique 'Nouvelle mission' pour démarrer.")
        st.stop()

    # Start timer when a scene is displayed and no previous judge exists
    if st.session_state.last_judge is None and state._active_started_at is None:
        state.start_attempt_timer()

    # Render mission
    st.divider()
    render_scene(scene, state)

    # Answer input
    st.divider()
    st.markdown("### 🎮 À toi de jouer")
    answer = render_answer_input(scene)
    st.session_state.answer = answer

    # Submit button
    submit_col1, submit_col2 = st.columns([1, 2])
    with submit_col1:
        if st.button("✅ Valider ma réponse"):
            submit_answer(state)
            st.rerun()

    with submit_col2:
        st.caption("Tu peux répondre même si ce n'est pas parfait — le jeu te corrige et t'apprend.")

    # Feedback
    if st.session_state.last_judge and st.session_state.last_engine_update:
        st.divider()
        st.markdown("## 🧾 Correction & Coaching")
        render_feedback(st.session_state.last_judge, st.session_state.last_engine_update, state)

        # Next mission CTA
        st.divider()
        if st.button("➡️ Mission suivante"):
            request_new_scene(state)
            st.rerun()

    st.divider()
    st.markdown('<div class="pq-footer">Python Quest • version V1 • by you + IA + Python 🔥</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()


