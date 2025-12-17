# game.py
"""
Python Quest — game core (state + rules + adaptive difficulty)

This module contains:
- Player profile/state (XP, level, streaks, inventory)
- Adaptive difficulty engine (auto up/down based on recent performance)
- Challenge tracking (topics, formats, error history)

UI (Streamlit) and AI (Gemini) are handled elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Tuple
import time
import math


# ----------------------------
# Enums
# ----------------------------

class Verdict(str, Enum):
    CORRECT = "correct"
    CLOSE = "close"
    WRONG = "wrong"


class ChallengeType(str, Enum):
    CODE_BUILDER = "code_builder"   # write code (priority)
    DEBUG_ARENA = "debug_arena"     # fix broken code
    QCM = "qcm"                     # multiple choice
    TRUE_FALSE = "true_false"       # concept check
    BOSS_FIGHT = "boss_fight"       # narrative boss challenge (often code)


class Topic(str, Enum):
    BASICS = "basics"
    DATA_STRUCTURES = "data_structures"
    CONTROL_FLOW = "control_flow"
    FUNCTIONS = "functions"
    OOP = "oop"
    PANDAS = "pandas"
    MATPLOTLIB = "matplotlib"
    MIXED = "mixed"


# ----------------------------
# Core data models
# ----------------------------

@dataclass
class PerformanceSample:
    """A single completed attempt."""
    timestamp: float
    challenge_type: str
    topic: str
    difficulty: int
    verdict: str
    xp_delta: int
    seconds_spent: Optional[float] = None


@dataclass
class PlayerStats:
    """Aggregated stats, mostly for adaptive difficulty and UI."""
    total_attempts: int = 0
    correct: int = 0
    close: int = 0
    wrong: int = 0

    # topic -> (attempts, wrong)
    topic_attempts: Dict[str, int] = field(default_factory=dict)
    topic_wrongs: Dict[str, int] = field(default_factory=dict)

    # type -> attempts
    type_attempts: Dict[str, int] = field(default_factory=dict)

    def record(self, sample: PerformanceSample) -> None:
        self.total_attempts += 1
        if sample.verdict == Verdict.CORRECT.value:
            self.correct += 1
        elif sample.verdict == Verdict.CLOSE.value:
            self.close += 1
        else:
            self.wrong += 1

        self.topic_attempts[sample.topic] = self.topic_attempts.get(sample.topic, 0) + 1
        if sample.verdict == Verdict.WRONG.value:
            self.topic_wrongs[sample.topic] = self.topic_wrongs.get(sample.topic, 0) + 1
        else:
            # keep key present for UI consistency
            self.topic_wrongs.setdefault(sample.topic, self.topic_wrongs.get(sample.topic, 0))

        self.type_attempts[sample.challenge_type] = self.type_attempts.get(sample.challenge_type, 0) + 1

    def accuracy(self) -> float:
        if self.total_attempts == 0:
            return 0.0
        return self.correct / self.total_attempts

    def recent_accuracy(self, history: List[PerformanceSample], n: int = 10) -> float:
        if not history:
            return 0.0
        recent = history[-n:]
        if not recent:
            return 0.0
        correct = sum(1 for s in recent if s.verdict == Verdict.CORRECT.value)
        return correct / len(recent)


@dataclass
class PlayerState:
    """
    Player state (persisted).
    - difficulty: overall difficulty level (1..10)
    - zone: current world zone (narrative)
    """
    player_name: str = "Joris"
    xp: int = 0
    level: int = 1
    difficulty: int = 4  # start intermediate-ish
    zone: str = "Village des Bases"

    streak: int = 0
    best_streak: int = 0
    hearts: int = 3  # like lives (kid-friendly)
    max_hearts: int = 3

    inventory: List[str] = field(default_factory=list)
    badges: List[str] = field(default_factory=list)

    stats: PlayerStats = field(default_factory=PlayerStats)
    history: List[PerformanceSample] = field(default_factory=list)

    # for pacing & adaptation
    _last_prompted_topic: Optional[str] = None
    _last_challenge_type: Optional[str] = None

    # to measure time spent (optional)
    _active_started_at: Optional[float] = None

    def start_attempt_timer(self) -> None:
        self._active_started_at = time.time()

    def stop_attempt_timer(self) -> Optional[float]:
        if self._active_started_at is None:
            return None
        seconds = time.time() - self._active_started_at
        self._active_started_at = None
        return max(0.0, seconds)

    def to_dict(self) -> dict:
        """JSON-serializable dict for storage."""
        d = asdict(self)
        # remove private runtime fields
        d.pop("_active_started_at", None)
        return d

    @staticmethod
    def from_dict(d: dict) -> "PlayerState":
        """Rehydrate from stored dict."""
        # stats
        stats_dict = d.get("stats", {})
        stats = PlayerStats(
            total_attempts=stats_dict.get("total_attempts", 0),
            correct=stats_dict.get("correct", 0),
            close=stats_dict.get("close", 0),
            wrong=stats_dict.get("wrong", 0),
            topic_attempts=stats_dict.get("topic_attempts", {}) or {},
            topic_wrongs=stats_dict.get("topic_wrongs", {}) or {},
            type_attempts=stats_dict.get("type_attempts", {}) or {},
        )

        # history
        hist = []
        for s in d.get("history", []) or []:
            hist.append(
                PerformanceSample(
                    timestamp=float(s.get("timestamp", time.time())),
                    challenge_type=str(s.get("challenge_type", "mixed")),
                    topic=str(s.get("topic", "mixed")),
                    difficulty=int(s.get("difficulty", 1)),
                    verdict=str(s.get("verdict", Verdict.WRONG.value)),
                    xp_delta=int(s.get("xp_delta", 0)),
                    seconds_spent=s.get("seconds_spent", None),
                )
            )

        state = PlayerState(
            player_name=d.get("player_name", "Joris"),
            xp=int(d.get("xp", 0)),
            level=int(d.get("level", 1)),
            difficulty=int(d.get("difficulty", 4)),
            zone=d.get("zone", "Village des Bases"),
            streak=int(d.get("streak", 0)),
            best_streak=int(d.get("best_streak", 0)),
            hearts=int(d.get("hearts", 3)),
            max_hearts=int(d.get("max_hearts", 3)),
            inventory=list(d.get("inventory", []) or []),
            badges=list(d.get("badges", []) or []),
            stats=stats,
            history=hist,
        )
        state._last_prompted_topic = d.get("_last_prompted_topic", None)
        state._last_challenge_type = d.get("_last_challenge_type", None)
        return state


# ----------------------------
# Rules + progression helpers
# ----------------------------

def xp_for_next_level(level: int) -> int:
    """
    XP curve: gentle early, steeper later.
    Example:
      level 1->2: 100
      level 2->3: 140
      ...
    """
    base = 80
    growth = 20
    return base + growth * (level - 1) + int(10 * math.sqrt(max(1, level - 1)))


def compute_level_from_xp(xp: int) -> int:
    """Compute current level given total xp."""
    lvl = 1
    remaining = xp
    while remaining >= xp_for_next_level(lvl):
        remaining -= xp_for_next_level(lvl)
        lvl += 1
        if lvl > 99:
            break
    return lvl


def xp_progress_within_level(xp: int) -> Tuple[int, int, float]:
    """
    Returns (current_level, xp_into_level, pct_to_next).
    """
    lvl = 1
    remaining = xp
    while remaining >= xp_for_next_level(lvl):
        remaining -= xp_for_next_level(lvl)
        lvl += 1
        if lvl > 99:
            break
    need = xp_for_next_level(lvl)
    pct = 0.0 if need <= 0 else remaining / need
    return lvl, remaining, max(0.0, min(1.0, pct))


def clamp(n: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, n))


# ----------------------------
# Adaptive difficulty engine
# ----------------------------

@dataclass
class DifficultyPolicy:
    min_difficulty: int = 1
    max_difficulty: int = 10

    # streak thresholds
    raise_after_correct_streak: int = 3
    lower_after_wrong_streak: int = 2

    # close answers are “half-success”
    close_counts_as: float = 0.5

    # recent performance window
    recent_n: int = 8

    # how strong adjustments are
    step_up: int = 1
    step_down: int = 1


def suggest_next_challenge_type(state: PlayerState) -> ChallengeType:
    """
    Prioritize code writing, but keep variety:
    - 60% code_builder
    - 20% debug_arena
    - 10% boss_fight (when level higher)
    - 10% quick checks (qcm / true_false)
    Deterministic selection based on stats and last type (no randomness here),
    UI can still randomize if desired.
    """
    last = state._last_challenge_type

    # If player is struggling: prefer DEBUG (often teaches faster) + small code
    recent = state.history[-5:]
    recent_wrong = sum(1 for s in recent if s.verdict == Verdict.WRONG.value)
    if recent_wrong >= 3:
        return ChallengeType.DEBUG_ARENA if last != ChallengeType.DEBUG_ARENA.value else ChallengeType.CODE_BUILDER

    # Default: rotate to avoid boredom
    if last is None:
        return ChallengeType.CODE_BUILDER

    if last == ChallengeType.CODE_BUILDER.value:
        return ChallengeType.DEBUG_ARENA
    if last == ChallengeType.DEBUG_ARENA.value:
        # occasional quick check
        return ChallengeType.QCM if state.level < 4 else ChallengeType.CODE_BUILDER
    if last in (ChallengeType.QCM.value, ChallengeType.TRUE_FALSE.value):
        return ChallengeType.CODE_BUILDER

    # Boss fights become more frequent after level 4
    if state.level >= 4 and last != ChallengeType.BOSS_FIGHT.value:
        return ChallengeType.BOSS_FIGHT

    return ChallengeType.CODE_BUILDER


def suggest_next_topic(state: PlayerState) -> Topic:
    """
    Choose topic based on:
    - current zone
    - weak topics (higher wrong rate)
    - avoid repeating exact same topic too much
    """
    # Map zones -> primary topics
    zone_map = {
        "Village des Bases": Topic.BASICS,
        "Jardin des Structures": Topic.DATA_STRUCTURES,
        "Pont du Contrôle": Topic.CONTROL_FLOW,
        "Atelier des Fonctions": Topic.FUNCTIONS,
        "Tour POO": Topic.OOP,
        "Lab Data": Topic.PANDAS,
        "Galerie des Graphiques": Topic.MATPLOTLIB,
    }

    primary = zone_map.get(state.zone, Topic.MIXED)

    # Determine weakness: wrong_rate = wrongs / attempts (with smoothing)
    weaknesses: List[Tuple[float, str]] = []
    for t, a in state.stats.topic_attempts.items():
        if a <= 0:
            continue
        w = state.stats.topic_wrongs.get(t, 0)
        wrong_rate = (w + 1) / (a + 3)  # smoothing
        weaknesses.append((wrong_rate, t))
    weaknesses.sort(reverse=True)

    last_topic = state._last_prompted_topic

    # If there is a clear weakness, steer there occasionally
    if weaknesses and weaknesses[0][0] >= 0.45:
        weak_topic = weaknesses[0][1]
        if weak_topic != last_topic:
            return Topic(weak_topic) if weak_topic in Topic._value2member_map_ else Topic.MIXED

    # Default to zone primary; if repeating too much, mix
    if last_topic == primary.value:
        return Topic.MIXED if primary != Topic.MIXED else Topic.FUNCTIONS

    return primary


def apply_adaptive_difficulty(
    state: PlayerState,
    policy: DifficultyPolicy = DifficultyPolicy(),
) -> None:
    """
    Adjust difficulty based on recent performance and streaks.
    Called after each judged attempt.
    """
    history = state.history
    if not history:
        return

    # Recent score (correct=1, close=0.5, wrong=0)
    recent = history[-policy.recent_n:]
    score = 0.0
    for s in recent:
        if s.verdict == Verdict.CORRECT.value:
            score += 1.0
        elif s.verdict == Verdict.CLOSE.value:
            score += policy.close_counts_as
    recent_rate = score / len(recent)

    # Streak-based adjustments
    # (We store streak as number of consecutive correct answers only)
    # Lowering uses consecutive wrong answers in last k
    last_k = history[-policy.lower_after_wrong_streak:]
    wrong_streak = all(s.verdict == Verdict.WRONG.value for s in last_k) if len(last_k) == policy.lower_after_wrong_streak else False

    if state.streak >= policy.raise_after_correct_streak and recent_rate >= 0.70:
        state.difficulty = clamp(state.difficulty + policy.step_up, policy.min_difficulty, policy.max_difficulty)
        # reset streak bonus so it doesn't climb too fast
        state.streak = 0
        return

    if wrong_streak and recent_rate <= 0.45:
        state.difficulty = clamp(state.difficulty - policy.step_down, policy.min_difficulty, policy.max_difficulty)
        return


# ----------------------------
# XP, hearts, inventory
# ----------------------------

def xp_delta_for_verdict(verdict: str, difficulty: int) -> int:
    """
    XP gain depends on verdict and difficulty.
    """
    difficulty = clamp(int(difficulty), 1, 10)
    if verdict == Verdict.CORRECT.value:
        return 12 + 2 * difficulty
    if verdict == Verdict.CLOSE.value:
        return 6 + difficulty
    return 0


def hearts_delta_for_verdict(verdict: str) -> int:
    """
    Wrong answer: lose 1 heart.
    Close: no loss.
    Correct: small chance to regain heart is handled elsewhere (deterministic here).
    """
    if verdict == Verdict.WRONG.value:
        return -1
    return 0


def maybe_award_badges(state: PlayerState) -> List[str]:
    """
    Award simple achievements. Returns list of newly awarded badges.
    """
    new = []

    if state.stats.total_attempts >= 10 and "Départ lancé" not in state.badges:
        state.badges.append("Départ lancé")
        new.append("Départ lancé")

    if state.best_streak >= 5 and "Combo x5" not in state.badges:
        state.badges.append("Combo x5")
        new.append("Combo x5")

    # topic mastery: at least 8 attempts with low wrong rate
    for t, a in state.stats.topic_attempts.items():
        if a >= 8:
            w = state.stats.topic_wrongs.get(t, 0)
            if w <= 2:
                badge = f"Maîtrise: {t}"
                if badge not in state.badges:
                    state.badges.append(badge)
                    new.append(badge)

    return new


def maybe_regain_heart(state: PlayerState) -> bool:
    """
    Deterministic heart regain rule:
    - every 4 correct answers (not necessarily consecutive), regain 1 heart if below max.
    """
    if state.hearts >= state.max_hearts:
        return False
    if state.stats.correct > 0 and state.stats.correct % 4 == 0:
        state.hearts = min(state.max_hearts, state.hearts + 1)
        return True
    return False


# ----------------------------
# Main update entrypoint
# ----------------------------

def apply_judgement(
    state: PlayerState,
    verdict: str,
    challenge_type: str,
    topic: str,
    difficulty_used: Optional[int] = None,
    seconds_spent: Optional[float] = None,
) -> Dict[str, object]:
    """
    Update game state from a judged attempt.

    Returns a dict with:
    - xp_delta, hearts_delta
    - level_up (bool), new_level
    - difficulty (new global difficulty)
    - gained_heart (bool)
    - new_badges (list)
    """
    if verdict not in (Verdict.CORRECT.value, Verdict.CLOSE.value, Verdict.WRONG.value):
        verdict = Verdict.WRONG.value

    difficulty_used = state.difficulty if difficulty_used is None else int(difficulty_used)

    # streak management (only correct increases streak)
    if verdict == Verdict.CORRECT.value:
        state.streak += 1
        state.best_streak = max(state.best_streak, state.streak)
    else:
        # close and wrong break the "perfect" streak
        state.streak = 0

    # xp & hearts
    xp_delta = xp_delta_for_verdict(verdict, difficulty_used)
    hearts_delta = hearts_delta_for_verdict(verdict)

    state.xp += xp_delta
    state.hearts = clamp(state.hearts + hearts_delta, 0, state.max_hearts)

    old_level = state.level
    state.level = compute_level_from_xp(state.xp)
    level_up = state.level > old_level

    # record sample
    sample = PerformanceSample(
        timestamp=time.time(),
        challenge_type=str(challenge_type),
        topic=str(topic),
        difficulty=clamp(int(difficulty_used), 1, 10),
        verdict=str(verdict),
        xp_delta=int(xp_delta),
        seconds_spent=seconds_spent,
    )
    state.history.append(sample)
    state.stats.record(sample)

    # update last selections for variety
    state._last_prompted_topic = str(topic)
    state._last_challenge_type = str(challenge_type)

    # adapt difficulty AFTER recording
    apply_adaptive_difficulty(state)

    # achievements / regen
    gained_heart = maybe_regain_heart(state)
    new_badges = maybe_award_badges(state)

    return {
        "xp_delta": xp_delta,
        "hearts_delta": hearts_delta,
        "level_up": level_up,
        "new_level": state.level,
        "difficulty": state.difficulty,
        "gained_heart": gained_heart,
        "new_badges": new_badges,
    }


def is_game_over(state: PlayerState) -> bool:
    """Game over when hearts are 0."""
    return state.hearts <= 0


def set_zone_for_level(state: PlayerState) -> None:
    """
    Simple mapping to move across the world.
    UI can override for more story control.
    """
    lvl = state.level
    if lvl <= 2:
        state.zone = "Village des Bases"
    elif lvl <= 3:
        state.zone = "Jardin des Structures"
    elif lvl <= 4:
        state.zone = "Atelier des Fonctions"
    elif lvl <= 6:
        state.zone = "Tour POO"
    elif lvl <= 7:
        state.zone = "Lab Data"
    else:
        state.zone = "Galerie des Graphiques"


def recommended_next(state: PlayerState) -> Dict[str, str]:
    """
    What the AI/UI should generate next: topic + challenge type + difficulty.
    """
    # zone update can be done here or in app
    set_zone_for_level(state)
    ctype = suggest_next_challenge_type(state).value
    topic = suggest_next_topic(state).value
    return {
        "zone": state.zone,
        "topic": topic,
        "challenge_type": ctype,
        "difficulty": str(state.difficulty),
    }
