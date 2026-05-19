"""
MAIN Irish (Gaeilge) Automated Scoring System
==============================================
Implements two complementary approaches for scoring MAIN narratives in Irish:

  1. Rules-Based Scorer  — keyword/phrase matching per macrostructural component
  2. GABert Scorer       — zero-shot sentence classification using
                           DCU-NLP/bert-base-irish-cased-v1 (GABert)

Sections scored (matching the MAIN Irish-Gaeilge 2020 rubric exactly):
  A. Story Structure       A1-A16/A17 depending on story (0-17 pts)
  B. Structural Complexity counts of GAO, GA/GO, AO, G sequences per episode
  C. Internal State Terms  total IST token count across 6 sub-categories
  D. Comprehension         D1-D10 question-response scoring (0-10 pts)

Based on:
  - MAIN Irish-Gaeilge 2020 rubric (Bohnacker & Gagarina, ZASPiL 63)
  - Baumann, Eller & Gagarina (2024), BERT-based annotation of MAIN narratives

Usage (Google Colab):
    !pip install transformers torch
    # Narrative transcripts -> transcripts/
    # Comprehension Q&A    -> transcripts/ with suffix _comprehension.txt
    # Then: python main_irish_scorer.py
    # With GABert: python main_irish_scorer.py --bert
    # Demo: python main_irish_scorer.py --demo
"""

import re
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# =============================================================================
# SECTION 1: TEXT NORMALISATION
# =============================================================================

def normalize_text(text: str) -> str:
    """Lowercase and strip punctuation, preserving Irish fada vowels."""
    text = text.lower()
    text = re.sub(r"[^\w\saeiouAEIOUaeiouAEIOU\u00e1\u00e9\u00ed\u00f3\u00fa\u00c1\u00c9\u00cd\u00d3\u00da]", "", text)
    return text


def split_sentences(text: str) -> List[str]:
    """Split text into sentences for BERT-based scoring."""
    raw = re.split(r"[.!?]+", text)
    return [s.strip() for s in raw if s.strip()]


# =============================================================================
# SECTION 2: KEYWORD LEXICONS - SECTION A (Story Structure)
#
# Each story has 3 episodes (Ep1, Ep2, Ep3).
# Each episode dict has keys: setting, ist_ie, goal, attempt, outcome, reaction
# All values are frozensets of lowercase Irish phrases matched as substrings.
#
# To extend: add phrases to any frozenset. No other code changes needed.
# =============================================================================

# Shared temporal/spatial anchors used by A1 setting across all stories
TIME = frozenset({
    "la amhain", "fado", "fado fado", "uair amhain", "trath den saol",
    "maidin amhain", "trathnona amhain", "oiche amhain", "ar maidin",
    "la brea", "trath",
})

PLACE = frozenset({
    "ag an loch", "in aice le loch", "ar bhruach na habhann",
    "i ngar don uisce", "ag an gcladach", "sa phairc", "sa ghort",
    "ar an talamh", "sa choill", "i measc na gcrann", "faoin gcrann",
    "ar bhruach na locha", "sa gharrai", "i gcoill", "i ngairdin",
    "i ngarrai", "i bpairc", "i nead", "thuas i gcrann", "sa nead",
    "sa chrann",
})

# We also keep fada versions for matching after normalisation fails gracefully
TIME_FADA = frozenset({
    "lá amháin", "fadó", "fadó fadó", "uair amháin", "tráth den saol",
    "maidin amháin", "tráthnóna amháin", "oíche amháin", "ar maidin",
    "lá breá", "tráth",
})

PLACE_FADA = frozenset({
    "ag an loch", "in aice le loch", "ar bhruach na habhann",
    "i ngar don uisce", "ag an gcladach", "sa pháirc", "sa ghort",
    "ar an talamh", "sa choill", "i measc na gcrann", "faoin gcrann",
    "ar bhruach na locha", "sa gharraí", "i gcoill", "i ngairdín",
    "i ngarraí", "i bpáirc", "i nead", "thuas i gcrann", "sa nead",
    "sa chrann",
    # Added from dog transcript: outdoor/near-tree locations
    "taobh amuigh", "in aice leis an crann", "sa chlós",
    # Note: "amuigh" and "in aice leis an" intentionally excluded —
    # too short; cause false positives across episode boundaries
})


def _match_any(text: str, keywords: frozenset) -> bool:
    return any(kw in text for kw in keywords)


def _norm_match(raw_text: str, keywords_fada: frozenset) -> bool:
    """Match against both normalised and original fada text."""
    norm = normalize_text(raw_text)
    norm_kws = frozenset(normalize_text(k) for k in keywords_fada)
    return _match_any(raw_text.lower(), keywords_fada) or _match_any(norm, norm_kws)


# -----------------------------------------------------------------------------
# DOG story (An Madra agus an Liathroid)
# Ep1: dog loses ball; Ep2: cat helps; Ep3: home together
# -----------------------------------------------------------------------------

DOG_A = {
    "setting": PLACE_FADA | TIME_FADA | frozenset({
        # "bhí an madra" removed — too generic, fires on every dog narrative
        "bhí liathróid", "ag an bpáirc", "sa pháirc",
        "taobh amuigh", "in aice leis an crann",
    }),
    "ist_ie": frozenset({
        "chonaic an madra liathróid", "thug an madra faoi deara liathróid",
        "bhí an madra ag breathnú ar an liathróid",
        "tháinig an liathróid", "chonaic sé liathróid",
        # Subject-specific forms only to avoid cross-episode firing
        "chonaic sé an luch", "chonaic an madra an luch",
        "bhí an luch", "suí in aice", "thug sé faoi deara",
    }),
    "goal": frozenset({
        "bhí an madra ag iarraidh breith ar an liathróid",
        "ag iarraidh an liathróid a fháil",
        "dul sa tóir ar an liathróid",
        "shocraigh an madra",
        "theastaigh ón madra an liathróid",
        # Added: catch/grab goal; also CHAT self-repair form from transcript
        "ag iarraidh greim air", "ag iarraidh greim a fháil",
        "ag iarraidh breith air",
        "ag iarraidh ag fháil greim",  # CHAT repair: "ag iarraidh ag fháil greim air"
        "ag fháil greim",
        "ag iarraidh an luch", "ag iarraidh é a fháil",
    }),
    "attempt": frozenset({
        "rith an madra", "léim an madra", "sa tóir ar an liathróid",
        "chuaigh an madra sa tóir", "lean sé an liathróid",
        "léim sé", "rith sé", "chuaigh sé sa tóir", "i ndiaidh",
    }),
    "outcome": frozenset({
        "thit an liathróid isteach san abhainn", "chuaigh an liathróid sa lochán",
        "ní bhfuair an madra an liathróid", "isteach san uisce",
        "d'imigh an liathróid", "theip air", "sa abhainn",
        # Added: dog hits tree or prey escapes
        "bhuail sé fhéin", "bhuail an madra", "bhuail sé in aghaidh",
        "in aghaidh an chrainn", "in aghaidh an chrann",
        "rith an luch uaidh", "d'éalaigh an luch",
        # Added: dog's head hits tree (CHAT phrasing: "chuaigh a cloigeann ag an crann")
        "chuaigh a cloigeann", "cloigeann ag an crann", "bhuail a cloigeann",
        "chuaigh sé ag an crann",
    }),
    "reaction": frozenset({
        "bhí díomá ar an madra", "bhí sé brónach", "bhí fearg air",
        "bhí an madra trína chéile", "brón", "díomá",
        # Added: injury as reaction after failed attempt
        "gortaithe", "bhí sé gortaithe", "bhí an madra gortaithe",
        "bhí pian air", "bhí sé tinn",
    }),
}

DOG_B = {
    "setting": frozenset({
        # Tightened: removed "bhí an madra" (too generic — fires in every episode)
        "ag an abhainn", "in aice leis an abhainn",
        "cois abhann", "taobh leis an abhainn",
        "bhí an cat agus an madra", "bhí an madra agus an cat",
    }),
    "ist_ie": frozenset({
        # Canonical Ep2 characters: cat notices dog's predicament
        "chonaic an cat", "thug an cat faoi deara", "chuala an cat",
        "tháinig an cat", "chonaic an cat go raibh an madra i gcruachás",
        "bhí duine ag tíocht", "tháinig duine",
        # Added: boy/person arrives from shop; dog reacts to balloon situation
        "tháinig buachaill", "tháinig an buachaill", "bhí buachaill ag tíocht",
        "tháinig sé ón siopa", "tháinig duine ón siopa",
        # Boy sad/distressed when balloon goes in tree (IST initiating Ep2)
        "bhí an buachaill brónach", "bhí sé brónach",
        "léim an balún sa chrann", "chuaigh an balún sa chrann",
        "léim an balún ag an crann", "bhí an balún sa chrann",
    }),
    "goal": frozenset({
        "bhí an cat ag iarraidh cabhrú leis an madra",
        "shocraigh an cat cabhrú",
        "ag iarraidh an liathróid a fháil",
        "ag iarraidh cabhrú",
        "theastaigh ón gcat cabhrú",
        "chun cabhrú leis an madra",
        "ag iarraidh an balún ar ais", "ag iarraidh an balún",
        "ag iarraidh é a fháil ar ais",
    }),
    "attempt": frozenset({
        "chuaigh an cat isteach san uisce", "léim an cat isteach",
        "thosaigh an cat ag snámh", "snámh an cat",
        "chuaigh an cat ag snámh",
        "ag tarraingt an balún", "ag baint an balún",
        "ag iarraidh an balún a bhaint",
    }),
    "outcome": frozenset({
        "fuair an cat an liathróid", "rug an cat ar an liathróid",
        "d'éirigh leis an gcat", "tháinig an cat ar ais leis an liathróid",
        "ní bhfuair an cat an liathróid", "theip ar an gcat",
        "bhí an cat fliuch",
        "lig sé leis an balún", "scaoil sé an balún",
        "chuaigh an balún sa chrann", "balún san crann",
        "ag an balún san crann",
        # Added: boy gets balloon back (Ep2 outcome in this story version)
        "fuair an buachaill a bhalún", "fuair sé an balún",
        "fuair sé a bhalún ar ais", "bhí an balún aige",
        "bhí an balún aici", "bhí an balún aci",
    }),
    "reaction": frozenset({
        "bhí an madra buíoch", "bhí áthas ar an madra",
        "bhí áthas ar an gcat", "bhí siad sásta", "buíoch",
        "bhí scanradh air", "bhí eagla ar an duine",
        "bhí sé scanraithe", "bhí eagla air agus",
        # Added: boy happy/pleased when he gets balloon back
        "bhí an buachaill sásta", "bhí sé sásta",
        "beagáinín sásta", "bhí an buachaill beagáinín sásta",
        "bhí áthas ar an mbuachaill", "bhí sé an-sásta",
    }),
}

DOG_C = {
    "setting": frozenset({
        "bhí siad", "ag an abhainn", "sa pháirc", "bhí an madra agus an cat",
    }),
    "ist_ie": frozenset({
        "bhí an madra fliuch", "bhí an cat fliuch",
        "bhí tuirse orthu", "chonaic siad", "bhí siad fuar",
        # Added: dog hungry / sees/looks at sausages (dog transcript Ep3)
        "bhí ocras ar an madra", "bhí ocras air",
        "chonaic an madra na hispíní", "chonaic sé na hispíní",
        "chonaic an madra ispín", "bhí ispíní ann",
        # Added: dog looks at sausages (perceptual IST as initiating event)
        "breathnaigh an madra ag an ispíní", "breathnaigh sé ag an ispíní",
        "bhí an madra ag breathnú ar na hispíní",
        "breathnaigh an madra ar na hispíní", "breathnaigh sé ar na hispíní",
        "bhí ispíní", "bhí na ispíní",
    }),
    "goal": frozenset({
        "ag iarraidh dul abhaile", "ag iarraidh triomú",
        "shocraigh siad dul abhaile", "theastaigh uathu dul abhaile",
        # Added: dog wants sausages (dog transcript Ep3)
        "ag iarraidh ispín", "ag iarraidh na hispíní",
        "ag iarraidh na hispíní a fháil", "ag iarraidh iad a fháil",
        "bhí ocras air", "theastaigh na hispíní",
        # Added: "ag iarraidh na ispíní a ithe" (no h — common transcription)
        "ag iarraidh na ispíní", "ag iarraidh ispíní a ithe",
        "ag iarraidh na ispíní a ithe", "ag iarraidh ispíní",
    }),
    "attempt": frozenset({
        "chuaigh siad abhaile", "rith siad abhaile",
        "d'imigh siad abhaile", "thosaigh siad ag rith",
        # Added: dog steals/eats sausages
        "sciob an madra", "ghoid an madra", "rug an madra ar na hispíní",
        "d'ith an madra", "thóg an madra na hispíní",
    }),
    "outcome": frozenset({
        "tháinig siad abhaile", "d'éirigh leo dul abhaile",
        "bhí siad tirim", "bhí siad sábháilte", "bhí siad sa bhaile",
        # Added: dog eats/gets sausages (dog transcript Ep3, line 44, 87)
        "bhí an madra ithe ispín", "d'ith sé na hispíní",
        "d'ith sé an ispín", "ithe ispín", "ithe na hispíní",
        "fuair sé an balún ar ais", "fuair an buachaill an balún",
        "chuaigh an buachaill an balún as",
        # Added: "ith an madra... na ispíní" (no d' prefix — CHAT repair form)
        "ith an madra", "ith sé na ispíní", "ith sé na hispíní",
        "ceann amháin acu na ispíní", "ceann amháin na ispíní",
        "ceann amháin acu na hispíní",
    }),
    "reaction": frozenset({
        "bhí siad sásta", "bhí áthas orthu", "bhí faoiseamh orthu",
        # bare "sásta" and "áthas" removed — fire on Ep2 text ("beagáinín sásta")
        # boy angry / dog satisfied — subject-anchored only
        "crosta", "bhí sé crosta", "bhí fearg air",
        "bhí an buachaill crosta", "bhí fearg ar an mbuachaill",
        "bhí an madra sásta", "bhí an buachaill sásta",
    }),
}


# =============================================================================
# SECTION 3: SECTION A - STORY STRUCTURE SCORER
# =============================================================================

def match_any(text: str, keywords: frozenset) -> bool:
    return any(kw in text for kw in keywords)


def score_setting(raw_text: str, episode_kws: dict,
                  ep_only_mode: bool = False) -> int:
    """
    A1 Setting scored 0-2:
      2 = time AND place reference both present
      1 = only one present
      0 = neither

    ep_only_mode=True (used for Ep2/Ep3):
      Only checks episode-specific setting terms, NOT the global TIME/PLACE sets.
      This prevents place/time phrases from Ep1 bleeding into later episode scores
      when the scorer applies all dicts to the full transcript text.
    """
    t = raw_text.lower()
    ep_only = episode_kws["setting"] - TIME_FADA - PLACE_FADA

    if ep_only_mode:
        # For Ep2/Ep3: only credit setting if an episode-specific marker appears
        return 1 if match_any(t, ep_only) else 0

    # For Ep1 (A1): check global TIME and PLACE sets + episode-specific
    has_time     = match_any(t, TIME_FADA)
    has_place    = match_any(t, PLACE_FADA)
    has_ep_place = match_any(t, ep_only)

    if has_time and (has_place or has_ep_place):
        return 2
    elif has_time or has_place or has_ep_place:
        return 1
    return 0


def score_episode_A(raw_text: str, episode_kws: dict,
                    ep_only_mode: bool = False) -> Dict[str, int]:
    """Score one episode for Section A."""
    t = raw_text.lower()
    scores = {"setting": score_setting(raw_text, episode_kws, ep_only_mode)}
    for comp in COMPONENTS[1:]:
        scores[comp] = 1 if match_any(t, episode_kws[comp]) else 0
    return scores


def score_section_A(text: str, story: str) -> dict:
    """Score all three episodes for Section A."""
    if story not in STORIES:
        raise ValueError(f"Unknown story '{story}'. Choose from: {list(STORIES)}")
    results = {}
    grand_total = 0
    for i, ep_kws in enumerate(STORIES[story]):
        ep_label = f"Ep{i+1}"
        # Ep1 uses global TIME/PLACE for A1 setting; Ep2/Ep3 use episode-specific only
        ep_only_mode = (i > 0)
        ep_scores = score_episode_A(text, ep_kws, ep_only_mode)
        ep_total = sum(ep_scores.values())
        results[ep_label] = {**ep_scores, "Total": ep_total}
        grand_total += ep_total
    results["Story_Total"]  = grand_total
    results["Max_Possible"] = 7 * 3   # setting(max 2) + 5 binary, times 3 episodes
    return results


# =============================================================================
# SECTION 4: SECTION B - STRUCTURAL COMPLEXITY
#
# For each episode, classify which GAO sequence type is present:
#   GAO  Goal + Attempt + Outcome  (highest complexity, B4)
#   GA   Goal + Attempt, no Outcome (B3)
#   GO   Goal + Outcome, no Attempt (B3)
#   AO   Attempt + Outcome, no Goal (B1)
#   G    Goal only (B2)
#   none none of G/A/O
#
# B1 = count of AO sequences  B2 = count of G-only  B3 = GA/GO  B4 = GAO
# =============================================================================

def classify_gao_sequence(ep_scores: dict) -> str:
    g = ep_scores.get("goal",    0) == 1
    a = ep_scores.get("attempt", 0) == 1
    o = ep_scores.get("outcome", 0) == 1
    if g and a and o:   return "GAO"
    if g and a:         return "GA"
    if g and o:         return "GO"
    if a and o:         return "AO"
    if g:               return "G"
    if a or o:          return "AO"   # A or O present without G still counts as AO
    return "none"


def score_section_B(section_a_result: dict) -> dict:
    """Derive Section B counts from Section A scores."""
    b = {"B1_AO": 0, "B2_G_only": 0, "B3_GA_GO": 0, "B4_GAO": 0, "episodes": {}}
    for ep_label in ["Ep1", "Ep2", "Ep3"]:
        ep = section_a_result.get(ep_label, {})
        seq = classify_gao_sequence(ep)
        b["episodes"][ep_label] = seq
        if seq == "GAO":      b["B4_GAO"]   += 1
        elif seq in ("GA","GO"): b["B3_GA_GO"] += 1
        elif seq == "AO":     b["B1_AO"]    += 1
        elif seq == "G":      b["B2_G_only"] += 1
    return b


# =============================================================================
# SECTION 5: SECTION C - INTERNAL STATE TERMS (IST)
#
# Count total IST tokens across 6 sub-categories (rubric C1).
# Each occurrence of a pattern is counted separately.
#
# Sub-categories:
#  1. Perceptual     feic, clois/cluin, mothaigh/braith/airigh, bolaigh
#  2. Physiological  tart, ocras, tuirse, gortaithe/tinn
#  3. Consciousness  beo, i do dhúiseacht, i do chodladh
#  4. Emotion        brón, sásta, áthas, fearg, imní, díomá, eagla, bród...
#  5. Mental verbs   ag iarraidh, smaoinigh, cuimhnigh, socraigh, creid...
#  6. Linguistic     abair, glaoigh, béic, scread, rabhadh, iarr
# =============================================================================

IST_PATTERNS: Dict[str, List[str]] = {
    "perceptual": [
        r"\bchonaic\b", r"\bfeiceann\b", r"\bd'fh[eé]ach\b", r"\bd'fheic\b",
        r"\bchlos\b", r"\bchuala\b", r"\bcloisim\b", r"\bcluineann\b",
        r"\bmhothaigh\b", r"\bmothaíonn\b", r"\bbraitheann\b", r"\bd'airigh\b",
        r"\bbolaigh\b",
        # Added: breathnaigh/breathnaíonn (to look at) — perceptual verb
        r"\bbreathnai\w*\b",
    ],
    "physiological": [
        r"\btart\b", r"\bocras\b", r"\btuirse\b", r"\btraochta\b",
        r"\bgortaithe\b", r"\btinn\b",
        # Added: "ceart go leor" as a physical-state IST (okay / not okay)
        r"\bceart go leor\b",
    ],
    "consciousness": [
        r"\bbeo\b", r"\bi do dh[uú]iseacht\b", r"\bm[uú]scailte\b",
        r"\bi do chodladh\b", r"\bina chodladh\b", r"\bina codladh\b",
    ],
    "emotion": [
        r"\bbrón\b", r"\bbrónach\b", r"\bsásta\b", r"\báthas\b",
        r"\bfearg\b", r"\bcrosta\b", r"\bimní\b", r"\bdíomá\b",
        r"\bdíomách\b", r"\beagla\b", r"\bbród\b", r"\bbródúil\b",
        r"\bcrógrach\b", r"\bslán\b", r"\bsábháilte\b", r"\bmíshásta\b",
        r"\bscanraithe\b", r"\bscanrúil\b", r"\bscanraigh\b", r"\bscantraigh\b",
        r"\biontas\b", r"\bionadh\b",
        r"\buaigneach\b", r"\bsona\b", r"\bfaoiseamh\b", r"\baerach\b",
    ],
    "mental_verbs": [
        r"\bag iarraidh\b", r"\bsmaoinigh\b", r"\bsmaoiníonn\b",
        r"\bcuimhnigh\b", r"\bcuimhníonn\b", r"\bfios\b",
        r"\bdearmad\b", r"\bshocraigh\b", r"\bsocraíonn\b",
        r"\bcreid\b", r"\bcreideann\b", r"\bar intinn\b",
        r"\bplean\b", r"\bsmaoineamh\b",
    ],
    "linguistic_verbs": [
        r"\bdúirt\b", r"\bdeir\b", r"\babair\b", r"\bglaoigh\b",
        r"\bghlaoigh\b", r"\bbéic\b", r"\bbéiceadh\b",
        r"\bscread\b", r"\bscreadaigh\b", r"\brabhadh\b",
        r"\bd'iarr\b", r"\biarrann\b",
    ],
}


def score_section_C(text: str) -> dict:
    """Count IST tokens. Returns per-sub-category counts and C1 total."""
    counts: Dict[str, int] = {}
    total = 0
    for category, patterns in IST_PATTERNS.items():
        cat_count = sum(len(re.findall(p, text, re.IGNORECASE))
                        for p in patterns)
        counts[category] = cat_count
        total += cat_count
    counts["C1_Total_IST"] = total
    return counts


# =============================================================================
# SECTION 6: SECTION D - COMPREHENSION SCORING
#
# Input: dict of {question_label: response_text}
# Parsed from a _comprehension.txt file with lines like:
#     D1: ag iarraidh bia a fháil do na héiníní
#     D2: tá ocras orthu
#
# Conditional questions (D3/D6/D9):
#   If the preceding question (D2/D5/D8) was answered correctly AND the
#   response contained an explanation/rationale, the follow-up is auto-scored
#   as 1 and need not be asked. The scorer implements this automatically.
# =============================================================================


# Dog story comprehension rubric
# Questions are based on the MAIN Irish dog story scoring sheet.
# The dog story used in the corpus features a dog, a boy, and sausages
# rather than the canonical dog/ball/lake — so correct answers reflect
# both the canonical rubric and the child's own story rendering.
D_DOG = {
    "D1": {
        # Why does the dog jump/run forward? (Goal Ep1)
        "correct": frozenset({
            "ag iarraidh an luch", "ag iarraidh breith ar an luch",
            "ag iarraidh greim a fháil", "bhí sé ag iarraidh",
            "ag iarraidh an liathróid", "teastaigh uaidh an luch",
            "rith sé ina dhiaidh", "bhí sé ag dul sa tóir",
        }),
        "wrong": frozenset({
            "le spraoi", "ag rith", "níl a fhios agam",
        }),
    },
    "D2": {
        # How does the dog feel? (IST/Reaction Ep1)
        "correct": frozenset({
            "gortaithe", "go dona", "tinn", "bhí sé gortaithe",
            "bhí pian air", "brónach", "díomách",
            # Added: "ní raibh sé ceart go leor" = not well/okay
            "ní raibh sé ceart go leor", "ní raibh sé go maith",
            "ní raibh sé i gceart",
        }),
        "wrong": frozenset({
            "go maith", "sásta", "áthas", "spraíúil",
        }),
        "has_explanation": frozenset({
            "bhuail sé fhéin", "in aghaidh an chrainn", "in aghaidh an chrann",
            "bhuail sé in aghaidh",
        }),
    },
    "D3": {
        # Why is the dog hurt? (follow-up; auto-scored if D2 has explanation)
        "correct": frozenset({
            "bhuail sé fhéin in aghaidh an chrainn",
            "bhuail sé in aghaidh an chrann",
            "rith sé isteach sa chrann",
            "chuaigh sé sa chrann", "in aghaidh an chrann",
            "in aghaidh an chloigeann",
            # Added: "gortaithe" as standalone cause; head-hitting phrasing
            "gortaithe", "bhí sé gortaithe",
            "chuaigh a cloigeann ag an crann", "bhuail a cloigeann",
            "cloigeann ag an crann",
        }),
        "wrong": frozenset({
            "thit sé", "d'éalaigh an luch", "níl a fhios agam",
        }),
    },
    "D4": {
        # Why does the boy jump up? (Goal Ep2)
        "correct": frozenset({
            "ag iarraidh an balún ar ais", "ag iarraidh an balún a fháil",
            "theastaigh an balún uaidh", "bhí sé ag iarraidh an balún",
            "ag iarraidh é a bhaint den chrann",
            "ag iarraidh an liathróid ar ais",
            # Added: possessive forms "ag iarraidh a balún a fháil"
            "ag iarraidh a balún a fháil", "ag iarraidh a bhalún a fháil",
            "ag iarraidh a bhalún", "ag iarraidh a balún",
        }),
        "wrong": frozenset({
            "le spraoi", "ag léim", "níl a fhios agam",
        }),
    },
    "D5": {
        # How does the boy feel? (IST/Reaction Ep2)
        "correct": frozenset({
            "sásta", "áthas", "go maith", "go breá",
            "bhí áthas air", "bhí sé sásta", "bhí sé lúcháireach",
        }),
        "wrong": frozenset({
            "go dona", "brónach", "feargach", "díomách",
        }),
        "has_explanation": frozenset({
            "fuair sé an balún ar ais", "fuair sé an balún",
            "bhain sé den chrann é",
        }),
    },
    "D6": {
        # Why does the boy feel good? (follow-up; auto-scored if D5 has explanation)
        "correct": frozenset({
            "fuair sé an balún ar ais", "fuair sé an balún",
            "tháinig an balún ar ais chuige", "bhain sé den chrann é",
            "d'éirigh leis an balún a fháil",
            # Added: possession forms "bhí an balún aici/aige/acu"
            "bhí an balún aige", "bhí an balún aici", "bhí an balún aci",
            "bhí a bhalún aige", "bhí a bhalún aici",
            "fuair sé a bhalún", "fuair sé a bhalún ar ais",
        }),
        "wrong": frozenset({
            "bhí sé ag léim", "bhí sé ag rith", "níl a fhios agam",
        }),
    },
    "D7": {
        # Why does the dog steal the sausages? (IST/Goal Ep3)
        "correct": frozenset({
            "bhí ocras air", "bhí ocras ar an madra",
            "theastaigh na hispíní uaidh", "bhí sé ag iarraidh ithe",
            "bhí sé ocrach", "bhí sé ag iarraidh na hispíní",
        }),
        "wrong": frozenset({
            "le spraoi", "bhí sé ag goid", "níl a fhios agam",
        }),
    },
    "D8": {
        # How does the boy feel seeing the dog? (IST/Reaction Ep3)
        "correct": frozenset({
            "crosta", "feargach", "míshásta", "díomách",
            "bhí fearg air", "bhí sé crosta", "bhí sé míshásta",
            "bhí díomá air",
            # Added: "brónach" accepted by human grader for this question
            "brónach", "bhí sé brónach", "bhí brón air",
            "an-chrosta", "bhí sé an-chrosta",
        }),
        "wrong": frozenset({
            "sásta", "áthas", "go maith", "spraíúil",
        }),
        "has_explanation": frozenset({
            "d'ith sé na hispíní", "d'ith an madra na hispíní",
            "ghoid an madra na hispíní", "sciob an madra",
        }),
    },
    "D9": {
        # Why would the boy be cross? (follow-up; auto-scored if D8 has explanation)
        "correct": frozenset({
            "d'ith sé na hispíní", "d'ith an madra na hispíní",
            "ghoid an madra na hispíní", "sciob an madra na hispíní",
            "d'ith sé a chuid ispíní", "bhí a chuid ispíní ite",
            # Added: verbal-noun past "ithe" without d' prefix; ispíní without h
            "ithe an madra", "ithe sé na hispíní", "ithe sé na ispíní",
            "mar ithe an madra", "ith an madra na ispíní",
            "ith sé na ispíní", "ith sé na hispíní",
        }),
        "wrong": frozenset({
            "tá an chuma sin air", "níl a fhios agam",
        }),
    },
    "D10": {
        # Will the boy be friendly with the dog? (with reason)
        "correct": frozenset({
            "no", "ní bheidh", "ní dóigh liom", "ní cheapfainn",
            "d'ith sé na hispíní", "mar ghoid sé",
            "mar d'ith sé", "mar sciob sé",
        }),
        "wrong": frozenset({
            "beidh", "yes", "sea", "beidh siad cairdiúil",
        }),
    },
}

# Registry — add other stories' D dicts when their rubrics are available
COMPREHENSION: Dict[str, dict] = {
    "birds": D_BIRDS,
    "dog":   D_DOG,
    # "cat":   D_CAT,
    # "goats": D_GOATS,
}

# Conditional pairs: {follow_up_question: prior_question}
CONDITIONAL_D = {"D3": "D2", "D6": "D5", "D9": "D8"}


def parse_comprehension_file(filepath: str) -> Dict[str, str]:
    """
    Parse a comprehension response file.
    Format: one response per line, e.g.:
        D1: ag iarraidh bia a fháil do na héiníní
        D2: tá ocras orthu
    """
    responses: Dict[str, str] = {}
    path = Path(filepath)
    if not path.exists():
        return responses
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^(D\d+)\s*:\s*(.+)$", line, re.IGNORECASE)
        if m:
            responses[m.group(1).upper()] = m.group(2).strip()
    return responses


def _score_one_D(response: str, q_kws: dict) -> Tuple[int, bool]:
    """
    Score a single comprehension response.
    Returns (score, has_explanation).
    score = 1 if a correct keyword found and no wrong keyword dominates.
    has_explanation = True if the response also contains an explanation phrase,
    which triggers auto-scoring of the next conditional question.
    """
    t = response.lower()
    is_correct = match_any(t, q_kws.get("correct", frozenset()))
    is_wrong   = match_any(t, q_kws.get("wrong",   frozenset()))
    has_expl   = match_any(t, q_kws.get("has_explanation", frozenset()))
    if is_wrong and not is_correct:
        return 0, False
    if is_correct:
        return 1, has_expl
    return 0, False


def score_section_D(responses: Dict[str, str], story: str) -> dict:
    """
    Score Section D comprehension questions.

    Conditional auto-scoring logic:
      If D2 response was correct AND contained an explanation phrase,
      D3 is automatically scored as 1 (and the rubric says not to ask it).
      Same logic for D6 from D5, and D9 from D8.
    """
    if story not in COMPREHENSION:
        return {"error": f"No comprehension rubric available for story '{story}'",
                "D_Total": 0, "Max_Possible": 10}

    q_rubric = COMPREHENSION[story]
    scores: dict = {}
    prior_had_explanation: Dict[str, bool] = {}

    for q_num in range(1, 11):
        q_label = f"D{q_num}"
        if q_label not in q_rubric:
            continue

        prior = CONDITIONAL_D.get(q_label)
        if prior and prior_had_explanation.get(prior, False):
            scores[q_label] = {"score": 1, "note": f"auto-scored (explanation in {prior})"}
            continue

        response = responses.get(q_label, "")
        if not response:
            scores[q_label] = {"score": 0, "note": "no response recorded"}
            continue

        score, has_expl = _score_one_D(response, q_rubric[q_label])
        scores[q_label] = {"score": score, "note": "scored from response"}
        prior_had_explanation[q_label] = has_expl

    total = sum(v["score"] for v in scores.values() if isinstance(v, dict))
    scores["D_Total"]      = total
    scores["Max_Possible"] = 10
    return scores


# =============================================================================
# SECTION 7: GABert SCORER
# =============================================================================

PROTOTYPES = {
    "setting": [
        "Lá amháin bhí Mamaí Éan agus a cuid éiníní beaga ina gcuid nead sa chrann.",
        "Fadó fadó bhí madra ag an abhainn.",
        "Maidin amháin bhí gabhair sa ghort.",
        "Tráth den saol bhí cat sa pháirc.",
    ],
    "ist_ie": [
        "Bhí ocras ar na héiníní beaga agus bhí siad ag caoineadh.",
        "Chonaic an cat go raibh na héiníní beaga ina n-aonar sa nead.",
        "D'imigh an mháthair agus d'fhág sí na héin ina ndiaidh.",
        "Chonaic an madra go raibh an t-éan i gcontúirt.",
        "Thug an cat faoi deara go raibh bia ann.",
    ],
    "goal": [
        "Bhí Mamaí Éan ag iarraidh bia a fháil do na héiníní.",
        "Bhí an cat ag iarraidh na héiníní a ithe.",
        "Shocraigh an madra na héiníní a shábháil.",
        "Bhí an cat ag iarraidh breith ar an bhféileacán.",
    ],
    "attempt": [
        "D'eitil an Mhamaí Éan léi ag cuardach bia.",
        "Chuaigh an cat suas sa gcrann ag dreapadóireacht.",
        "Rug an madra greim ar eireaball an chait.",
        "Léim an cat i dtreo an fhéileacáin.",
    ],
    "outcome": [
        "Tháinig an Mhamaí Éan ar ais le péist do na héiníní.",
        "Ba bheag nár rug an cat ar cheann de na héiníní.",
        "Chuir an madra ruaig ar an gcat agus sábháladh na héiníní.",
        "Thit an cat isteach sa lochán.",
    ],
    "reaction": [
        "Bhí áthas ar na héiníní beaga.",
        "Bhí eagla mhór ar na héiníní agus bhí an cat sásta.",
        "Bhí an madra bródúil gur shábháil sé na héiníní.",
        "Bhí an cat crosta agus bhí díomá air.",
        "Bhí faoiseamh ar na héiníní.",
    ],
    "none": [
        "Agus ansin.", "Mar sin.", "Ceart go leor.", "Ar aon nós.",
    ],
}


def load_gabert():
    """Load GABert. Requires: pip install transformers torch"""
    try:
        from transformers import AutoTokenizer, AutoModel
        import torch
        model_name = "DCU-NLP/bert-base-irish-cased-v1"
        print(f"Loading GABert ({model_name}) ...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        return tokenizer, model, torch
    except ImportError:
        print("ERROR: pip install transformers torch")
        return None, None, None


def _mean_pool(token_emb, attn_mask):
    mask = attn_mask.unsqueeze(-1).expand(token_emb.size()).float()
    return __import__('torch').sum(token_emb * mask, 1) / \
           __import__('torch').clamp(mask.sum(1), min=1e-9)


def _embed(texts, tokenizer, model, torch):
    enc = tokenizer(texts, padding=True, truncation=True,
                    max_length=128, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc)
    return _mean_pool(out.last_hidden_state, enc["attention_mask"])


def _cosine(a, b):
    a = a / a.norm(dim=-1, keepdim=True)
    b = b / b.norm(dim=-1, keepdim=True)
    return a @ b.T


def score_section_A_gabert(text: str, story: str,
                            tokenizer, model, torch,
                            threshold: float = 0.60) -> dict:
    """
    Section A scoring using GABert zero-shot sentence classification.
    Sentences split into thirds to approximate episode boundaries.
    """
    proto_embs = {lbl: _embed(sents, tokenizer, model, torch)
                  for lbl, sents in PROTOTYPES.items()}
    sents = split_sentences(text)
    n = len(sents)
    ep_splits = [sents[:n//3], sents[n//3:2*n//3], sents[2*n//3:]]

    results: dict = {}
    grand_total = 0

    for i, ep_sents in enumerate(ep_splits):
        ep_label = f"Ep{i+1}"
        found = {c: False for c in COMPONENTS}
        sent_log = []

        for sent in ep_sents:
            if not sent.strip():
                continue
            s_emb = _embed([sent], tokenizer, model, torch)
            best_lbl, best_score = "none", -1.0
            for lbl, p_emb in proto_embs.items():
                score = _cosine(s_emb, p_emb).max().item()
                if score > best_score:
                    best_score, best_lbl = score, lbl
            sent_log.append((sent, best_lbl, round(best_score, 3)))
            if best_score >= threshold and best_lbl in found:
                found[best_lbl] = True

        ep_scores = {c: (1 if found[c] else 0) for c in COMPONENTS}
        # Setting: 2 if setting-type sentences found; can't distinguish
        # time vs place from embeddings alone without finer-grained prototypes
        ep_scores["setting"] = 2 if found["setting"] else 0
        ep_total = sum(ep_scores.values())
        results[ep_label] = {**ep_scores, "Total": ep_total,
                              "_sentences": sent_log}
        grand_total += ep_total

    results["Story_Total"]  = grand_total
    results["Max_Possible"] = 21
    return results


# =============================================================================
# SECTION 8: REPORTING & EXPORT
# =============================================================================

def print_full_report(transcript_id: str, story: str,
                      sec_a: dict, sec_b: dict, sec_c: dict,
                      sec_d: Optional[dict] = None,
                      sec_a_bert: Optional[dict] = None):
    """Print a complete MAIN score report to stdout."""
    W = 66
    print(f"\n{'='*W}")
    print(f"  TRANSCRIPT: {transcript_id}  |  STORY: {story.upper()}")
    print(f"{'='*W}")

    # Section A
    print(f"\n  SECTION A - Story Structure  (max {sec_a.get('Max_Possible', 21)})")
    hdr = f"  {'Component':<30} {'Rules':>6}"
    if sec_a_bert:
        hdr += f"  {'BERT':>6}  {'':>5}"
    print(hdr)
    print(f"  {'-'*(W-2)}")

    for i in range(1, 4):
        ep_label = f"Ep{i}"
        ep_r = sec_a.get(ep_label, {})
        ep_b = (sec_a_bert or {}).get(ep_label, {})
        print(f"\n  Episode {i}")
        for comp in COMPONENTS:
            r = ep_r.get(comp, 0)
            line = f"  {COMPONENT_LABELS[comp]:<30} {r:>6}"
            if sec_a_bert:
                b = ep_b.get(comp, 0)
                agree = "OK" if r == b else "!!"
                line += f"  {b:>6}  {agree:>5}"
            print(line)
        print(f"  {'Episode Total':<30} {ep_r.get('Total', 0):>6}")

    print(f"\n  {'Story Total':<30} {sec_a.get('Story_Total', 0):>6}", end="")
    if sec_a_bert:
        print(f"  {sec_a_bert.get('Story_Total', 0):>6}", end="")
    print(f"\n  {'Max Possible':<30} {sec_a.get('Max_Possible', 21):>6}")

    # Section B
    print(f"\n  SECTION B - Structural Complexity")
    for key, label in [("B1_AO","B1 AO sequences"), ("B2_G_only","B2 G-only sequences"),
                        ("B3_GA_GO","B3 GA/GO sequences"), ("B4_GAO","B4 GAO sequences")]:
        print(f"  {label:<30} {sec_b.get(key, 0):>6}")
    for ep, seq in sec_b.get("episodes", {}).items():
        print(f"     {ep}: {seq}")

    # Section C
    print(f"\n  SECTION C - Internal State Terms")
    for cat in ["perceptual","physiological","consciousness",
                "emotion","mental_verbs","linguistic_verbs"]:
        print(f"  {cat:<30} {sec_c.get(cat, 0):>6}")
    print(f"  {'C1 Total IST':<30} {sec_c.get('C1_Total_IST', 0):>6}")

    # Section D
    if sec_d:
        print(f"\n  SECTION D - Comprehension  (max {sec_d.get('Max_Possible', 10)})")
        if "error" in sec_d:
            print(f"  {sec_d['error']}")
        else:
            for q_num in range(1, 11):
                q = f"D{q_num}"
                if q in sec_d:
                    val = sec_d[q]
                    score = val.get("score", 0) if isinstance(val, dict) else val
                    note  = val.get("note", "")  if isinstance(val, dict) else ""
                    print(f"  {q:<30} {score:>6}   {note}")
            print(f"  {'D Total':<30} {sec_d.get('D_Total', 0):>6}")

    print(f"\n{'='*W}\n")


def flatten_for_csv(transcript_id: str, story: str, method: str,
                    sec_a: dict, sec_b: dict, sec_c: dict,
                    sec_d: Optional[dict] = None) -> dict:
    """Flatten all section scores into one CSV row."""
    row: dict = {
        "transcript_id": transcript_id,
        "story":         story,
        "method":        method,
        "A_Story_Total": sec_a.get("Story_Total", 0),
        "A_Max":         sec_a.get("Max_Possible", 21),
    }
    for i in range(1, 4):
        ep_label = f"Ep{i}"
        ep = sec_a.get(ep_label, {})
        for comp in COMPONENTS:
            row[f"A_{ep_label}_{comp}"] = ep.get(comp, 0)
        row[f"A_{ep_label}_Total"] = ep.get("Total", 0)

    row.update({
        "B1_AO":     sec_b.get("B1_AO",    0),
        "B2_G":      sec_b.get("B2_G_only", 0),
        "B3_GA_GO":  sec_b.get("B3_GA_GO", 0),
        "B4_GAO":    sec_b.get("B4_GAO",   0),
        "B_Ep1_seq": sec_b.get("episodes", {}).get("Ep1", ""),
        "B_Ep2_seq": sec_b.get("episodes", {}).get("Ep2", ""),
        "B_Ep3_seq": sec_b.get("episodes", {}).get("Ep3", ""),
    })

    for cat in ["perceptual","physiological","consciousness",
                "emotion","mental_verbs","linguistic_verbs"]:
        row[f"C_{cat}"] = sec_c.get(cat, 0)
    row["C_Total_IST"] = sec_c.get("C1_Total_IST", 0)

    if sec_d and "error" not in sec_d:
        for q_num in range(1, 11):
            q = f"D{q_num}"
            val = sec_d.get(q, {})
            row[q] = val.get("score", 0) if isinstance(val, dict) else 0
        row["D_Total"] = sec_d.get("D_Total", 0)

    return row


def export_csv(rows: list, output_path: str):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved -> {output_path}")


# =============================================================================
# SECTION 9: CHAT FILE PARSER
#
# Handles CLAN/CHAT format transcripts (.cha files) as produced by CLAN software.
#
# CHAT conventions handled here:
#   *SPxx:   speaker turn lines (may wrap onto continuation lines)
#   %com %exp %err %pho etc.  dependent/comment tiers — excluded
#   &word    filled pauses / false starts          — removed
#   &+word   phonological fragment                 — removed
#   [/]      retracing marker                      — removed
#   [//]     reformulation marker                  — removed
#   [///]    reformulation with retrace            — removed
#   [?]      uncertain transcription               — removed
#   [=? x]   transcription alternative             — removed, keep original word
#   [: x]    expansion of preceding item           — keep expansion x, drop marker
#   [* x]    error coding                          — removed
#   xxx      unintelligible speech                 — removed
#   yyy      phonological coding                   — removed
#   www      untranscribed                         — removed
#   word@i   code suffix (e.g. ok@i = "okay" in English) — strip @... suffix
#   <text>   scope markers for annotations        — angle brackets removed, keep text
#   +...     special utterance linkers             — removed
#   0word    omitted word markers                 — removed
#
# Speaker identification:
#   The child speaker label is read from the @ID tier where the role field
#   contains "Target_Child". If not found, falls back to the --child-speaker
#   argument (default "SP02").
#
# Comprehension questions:
#   Examiner turns during the comprehension phase (after the narrative prompt)
#   are extracted into a separate comprehension response dict. The heuristic
#   is that the narrative phase ends when the examiner asks "ar thaithnigh an
#   scéal leat" or similar warm-up question (coded as D0 in the rubric).
# =============================================================================

# CHAT annotation patterns to strip from utterances
_CHAT_STRIP = [
    re.compile(r'\[=\?[^\]]*\]'),       # transcription alternatives [=? word] — remove whole tag
    re.compile(r'\[:[^\]]*\]'),          # expansions [: word] — remove marker, keep preceding word
    re.compile(r'\[[^\]]*\]'),           # all remaining bracket annotations [...]
    re.compile(r'<([^>]*)>'),            # angle-bracket scope markers — keep inner text
    re.compile(r'&\+\S+'),              # phonological fragments &+word
    re.compile(r'&\S+'),                # filled pauses / false starts &word
    re.compile(r'\b(xxx|yyy|www)\b'),   # unintelligible / untranscribed
    re.compile(r'\b0\w+'),              # omitted word markers 0word
    re.compile(r'\+[\/\\\.]{1,3}'),     # utterance linkers +//, +., etc.
    re.compile(r'@\w+'),                # code suffixes word@i, word@s etc.
    # Partial word markers: (text) indicates a partially articulated word
    # e.g. (a)g -> ag, b(e)cause -> because. Strip the parentheses, keep letters.
    re.compile(r'\((\w+)\)'),           # (text) partial markers -> keep inner text
    re.compile(r'\s{2,}'),              # collapse multiple spaces
]

# Phrases that mark the boundary between narrative and comprehension phases
_COMP_BOUNDARY_PHRASES = frozenset({
    "ar thaithnigh an scéal leat",
    "ar thaithin an scéal leat",    # fada-less variant
    "ar thaithnigh an sceal leat",  # no fada on scéal
    "ar thaithin an sceal leat",    # fully fada-less
    "an maith leat an scéal",
    "an maith leat an sceal",
    "an dtaitníonn an scéal",
    "an dtaitnionn an sceal",
    # "an bhfuil tú réidh" removed — SP01 uses this mid-narrative to check
    # the child is ready for the next picture set, causing premature phase switch
})
# Normalised (fada-stripped) versions — transcribers sometimes omit diacritics
# e.g. "ar thaithin" instead of "ar thaithnigh an sceal"
_FADA = str.maketrans("áéíóúÁÉÍÓÚ", "aeiouAEIOU")
_COMP_BOUNDARY_PHRASES_NORM = frozenset(
    ph.translate(_FADA) for ph in _COMP_BOUNDARY_PHRASES
)


def _is_boundary(clean_lower: str) -> bool:
    if any(ph in clean_lower for ph in _COMP_BOUNDARY_PHRASES):
        return True
    norm = clean_lower.translate(_FADA)
    return any(ph in norm for ph in _COMP_BOUNDARY_PHRASES_NORM)

# Primary question keywords — any of these alone flag a scored comprehension question.
_COMP_QUESTION_INTRO = frozenset({
    "cén fáth", "conas", "cén chaoi", "cé acu", "samhlaigh", "meas tú", "cad",
    "an mbeidh",   # D10 in all stories: "an mbeidh X cáirdiúil/sásta..."
})

# "lig ort" is only a question trigger when paired with a direct question word
# in the SAME turn — e.g. "lig ort féin... cén chaoi..."
# Standing alone it is a scene-setter ("pretend you can see the boy"), not a question.
_QUESTION_WORDS = frozenset({"cén", "conas", "cad", "an mbeidh", "an bhfuil"})


def _is_comp_question(clean_lower: str) -> bool:
    """Return True if an examiner turn introduces a new scored comp question."""
    if any(kw in clean_lower for kw in _COMP_QUESTION_INTRO):
        return True
    if "lig ort" in clean_lower:
        return any(qw in clean_lower for qw in _QUESTION_WORDS)
    return False


def _clean_chat_utterance(raw: str) -> str:
    """
    Strip CHAT annotations from a single utterance string.
    Returns clean orthographic text suitable for keyword matching.
    """
    text = raw
    for pattern in _CHAT_STRIP:
        if pattern.groups:          # patterns with capture groups (angle brackets)
            text = pattern.sub(r'\1', text)
        else:
            text = pattern.sub(' ', text)
    # Remove trailing punctuation CLAN adds (. ! ?)  but keep internal
    text = text.strip().rstrip('.!?')
    text = re.sub(r'\s{2,}', ' ', text).strip()
    return text


def _detect_child_speaker(lines: List[str]) -> str:
    """
    Read the @ID tiers to find which speaker code has role Target_Child.
    Falls back to 'SP02' if not found.

    @ID format:
        @ID: language|corpus|code|age|sex|group|SES|role|education|custom|
    The role field is index 7 (0-based).
    """
    for line in lines:
        if line.startswith("@ID:"):
            parts = line[4:].strip().split("|")
            if len(parts) > 7 and "Target_Child" in parts[7]:
                return parts[2].strip()   # speaker code is field index 2
    return "SP02"


def parse_chat_file(filepath: str,
                    child_speaker: Optional[str] = None
                    ) -> Tuple[str, Dict[str, str]]:
    """
    Parse a CLAN CHAT (.cha) file and return:
        narrative_text  — cleaned child speech from the narrative phase
        comp_responses  — dict of {D1..D10: response_text} from comprehension phase

    Parameters
    ----------
    filepath : str
        Path to the .cha file.
    child_speaker : str, optional
        Override the speaker code for the child. If None, auto-detected from @ID.

    Returns
    -------
    narrative_text : str
        Concatenated cleaned utterances from the child during the narrative phase.
    comp_responses : dict
        Comprehension responses keyed D1..D10, extracted from child turns
        during the comprehension phase.
    """
    path = Path(filepath)
    raw_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    # Auto-detect child speaker from @ID if not overridden
    if child_speaker is None:
        child_speaker = _detect_child_speaker(raw_lines)

    child_prefix   = f"*{child_speaker}:"
    examiner_codes = set()   # all speaker codes that are NOT the child

    # Collect all speaker codes from @Participants tier
    for line in raw_lines:
        if line.startswith("@Participants:"):
            # Format: @Participants: CODE Name Role, CODE Name Role, ...
            entries = line[14:].strip().split(",")
            for entry in entries:
                parts = entry.strip().split()
                if parts:
                    code = parts[0]
                    if code != child_speaker:
                        examiner_codes.add(f"*{code}:")

    # ── Pass 1: reconstruct full turns (handle CHAT continuation lines) ──────
    # In CHAT, continuation lines begin with a TAB and have no speaker prefix.
    # Dependent tiers (%com, %exp etc.) also begin with TAB+% and are excluded.
    turns: List[Tuple[str, str]] = []   # [(speaker_prefix, full_utterance)]
    current_speaker = None
    current_words: List[str] = []

    for line in raw_lines:
        if line.startswith("@"):
            # Header line — flush any open turn
            if current_speaker and current_words:
                turns.append((current_speaker, " ".join(current_words)))
            current_speaker = None
            current_words = []
            continue

        if line.startswith("*"):
            # New speaker turn
            if current_speaker and current_words:
                turns.append((current_speaker, " ".join(current_words)))
            colon_idx = line.index(":") if ":" in line else -1
            if colon_idx > 0:
                current_speaker = line[:colon_idx + 1]   # e.g. "*SP02:"
                current_words   = [line[colon_idx + 1:].strip()]
            else:
                current_speaker = None
                current_words   = []
            continue

        if line.startswith("\t"):
            content = line.strip()
            if content.startswith("%"):
                # Dependent tier — skip entirely
                continue
            if current_speaker:
                current_words.append(content)
            continue

    # Flush final turn
    if current_speaker and current_words:
        turns.append((current_speaker, " ".join(current_words)))

    # ── Pass 2: split narrative vs comprehension phase ────────────────────────
    # The narrative phase ends when the examiner produces a warm-up/closing
    # phrase (the D0 question). Everything before = narrative; after = comprehension.
    narrative_phase    = True
    comp_question_num  = 0           # tracks which D-question we're on
    in_comp_question   = False       # True immediately after an examiner D-question

    narrative_utterances: List[str] = []
    comp_responses: Dict[str, str]  = {}

    for speaker, utterance in turns:
        is_child    = speaker == child_prefix
        is_examiner = speaker in examiner_codes or (
            speaker != child_prefix and speaker.startswith("*"))

        clean = _clean_chat_utterance(utterance)
        if not clean:
            continue

        clean_lower = clean.lower()

        if narrative_phase:
            # Check if examiner has said the boundary phrase
            if is_examiner and _is_boundary(clean_lower):
                narrative_phase   = False
                in_comp_question  = False
                comp_question_num = 0
                continue

            if is_child:
                narrative_utterances.append(clean)

        else:
            # Comprehension phase
            if is_examiner:
                if _is_comp_question(clean_lower):
                    current_slot = f"D{comp_question_num + 1}"
                    if current_slot not in comp_responses:
                        # New primary question — advance counter and open slot.
                        comp_question_num += 1
                        in_comp_question = True
                    else:
                        # Slot already filled: this is a follow-up / clarifying
                        # prompt from the examiner.  Do NOT advance the counter
                        # or re-open the slot — the stored answer must not be
                        # overwritten by the child's next response.
                        in_comp_question = False
                else:
                    in_comp_question = False

            elif is_child and in_comp_question and comp_question_num <= 10:
                q_label = f"D{comp_question_num}"
                # Append if child gives multiple turns for same question
                if q_label in comp_responses:
                    comp_responses[q_label] += " " + clean
                else:
                    comp_responses[q_label] = clean
                in_comp_question = False   # one child response per question

    narrative_text = " ".join(narrative_utterances)
    return narrative_text, comp_responses


def _parse_chat_style_txt(lines: List[str],
                          child_speaker: str) -> Tuple[str, Dict[str, str]]:
    """
    Extract child-only speech from a plain .txt file that uses CHAT-style
    speaker prefixes (*SP01:, *SP02: etc.).

    SP02 (or whichever code is child_speaker) acts as an "opening bracket":
    text between a *SP02: line and the next *SP01: line belongs to the child.
    SP01 (or any non-child speaker) acts as a "closing bracket": their lines
    and any text that follows until the next *SP02: line is ignored.

    The narrative / comprehension split and D-label assignment follow the same
    logic used by parse_chat_file for .cha files.
    """
    child_prefix = f"*{child_speaker}:"

    # ── Reconstruct full turns (handle CHAT continuation lines) ─────────────
    turns: List[Tuple[str, str]] = []
    current_speaker: Optional[str] = None
    current_words: List[str] = []

    for line in lines:
        if line.startswith("@"):
            if current_speaker and current_words:
                turns.append((current_speaker, " ".join(current_words)))
            current_speaker = None
            current_words = []
            continue

        if line.startswith("*"):
            if current_speaker and current_words:
                turns.append((current_speaker, " ".join(current_words)))
            colon_idx = line.index(":") if ":" in line else -1
            if colon_idx > 0:
                current_speaker = line[:colon_idx + 1]
                current_words   = [line[colon_idx + 1:].strip()]
            else:
                current_speaker = None
                current_words   = []
            continue

        if line.startswith("\t") or (line and not line.startswith("*")):
            content = line.strip()
            if content.startswith("%"):
                continue          # skip dependent tiers
            if current_speaker and content:
                current_words.append(content)
            continue

    if current_speaker and current_words:
        turns.append((current_speaker, " ".join(current_words)))

    # ── Split narrative vs comprehension, collect child turns only ───────────
    narrative_phase   = True
    comp_question_num = 0
    in_comp_question  = False
    narrative_utterances: List[str] = []
    comp_responses: Dict[str, str]  = {}

    for speaker, utterance in turns:
        is_child    = speaker == child_prefix
        is_examiner = speaker != child_prefix and speaker.startswith("*")

        clean = _clean_chat_utterance(utterance)
        if not clean:
            continue
        clean_lower = clean.lower()

        if narrative_phase:
            if is_examiner and _is_boundary(clean_lower):
                narrative_phase   = False
                in_comp_question  = False
                comp_question_num = 0
                continue
            if is_child:
                narrative_utterances.append(clean)
        else:
            if is_examiner:
                if _is_comp_question(clean_lower):
                    current_slot = f"D{comp_question_num + 1}"
                    if current_slot not in comp_responses:
                        # New primary question — advance counter and open slot.
                        comp_question_num += 1
                        in_comp_question = True
                    else:
                        # Slot already filled: follow-up / clarifying prompt.
                        # Do NOT advance counter or re-open slot.
                        in_comp_question = False
                else:
                    in_comp_question = False
            elif is_child and in_comp_question and comp_question_num <= 10:
                q_label = f"D{comp_question_num}"
                if q_label in comp_responses:
                    comp_responses[q_label] += " " + clean
                else:
                    comp_responses[q_label] = clean
                in_comp_question = False

    return " ".join(narrative_utterances), comp_responses


def load_transcript(filepath: str,
                    child_speaker: Optional[str] = None
                    ) -> Tuple[str, Dict[str, str], str]:
    """
    Load a transcript from either a .cha (CHAT) or .txt file.

    For .cha files: runs the full CHAT parser (child-only, auto-detected speaker).
    For .txt files: if the file contains CHAT-style *SPxx: speaker lines the
                    file is treated identically to a .cha file — only the child's
                    turns are extracted and the narrative / comprehension split is
                    applied automatically.  Plain prose .txt files (no speaker
                    lines) are returned as-is, as before.

    Returns
    -------
    narrative_text : str
    comp_responses : dict   (may be empty for plain-prose .txt files)
    child_speaker  : str    (detected or supplied speaker code)
    """
    p = Path(filepath)
    raw_lines = p.read_text(encoding="utf-8", errors="replace").splitlines()

    if p.suffix.lower() == ".cha":
        narrative, comp = parse_chat_file(filepath, child_speaker)
        detected = child_speaker or _detect_child_speaker(raw_lines)
        return narrative, comp, detected

    # ── .txt branch ──────────────────────────────────────────────────────────
    # Detect whether the file uses CHAT-style *SPxx: speaker prefixes.
    has_speaker_lines = any(
        re.match(r'^\*[A-Z0-9]+:', line) for line in raw_lines
    )

    if has_speaker_lines:
        # Auto-detect child speaker the same way parse_chat_file does
        detected = child_speaker or _detect_child_speaker(raw_lines)
        # Fall back to SP02 if @ID tier is absent (common in .txt exports)
        if detected == "SP02" and child_speaker is None:
            # honour explicit override; otherwise keep SP02 as the default
            pass
        narrative, comp = _parse_chat_style_txt(raw_lines, detected)
        return narrative, comp, detected
    else:
        # Plain prose transcript — return full text unchanged
        return p.read_text(encoding="utf-8"), {}, "n/a"


# =============================================================================
# SECTION 10: BATCH PROCESSING
# =============================================================================

def infer_story(filename: str) -> Optional[str]:
    name = filename.lower()
    if "birds" in name or "eanlaith" in name or "ean" in name:
        return "birds"
    if "goats" in name or "gabhar" in name:
        return "goats"
    if "dog" in name or "madra" in name:
        return "dog"
    if "cat" in name:
        return "cat"
    return None


def process_transcripts(transcripts_dir: str = "transcripts",
                        use_bert: bool = False,
                        output_csv: str = "main_scores.csv",
                        child_speaker: Optional[str] = None,
                        story_filter: Optional[str] = None):
    """
    Batch-process all transcripts in transcripts_dir.

    Accepts both CHAT (.cha) and plain text (.txt) files.

    File naming conventions
    -----------------------
    Narrative .cha :   <id>_<story>.cha   e.g. P01_dog.cha
    Narrative .txt :   <id>_<story>.txt   e.g. P01_dog.txt
    Comprehension  :   <id>_<story>_comprehension.txt  (only needed for .txt files;
                       for .cha files comprehension is extracted automatically)

    Parameters
    ----------
    transcripts_dir : str
    use_bert        : bool
    output_csv      : str
    child_speaker   : str, optional
        Override child speaker code (e.g. "SP02"). If None, auto-detected
        from the @ID tier of each .cha file; ignored for .txt files.
    story_filter    : str, optional
        If set, only process files matching this story (e.g. "dog").
    """
    p = Path(transcripts_dir)
    if not p.exists():
        print(f"ERROR: directory '{transcripts_dir}' not found.")
        return

    # Collect .cha and .txt files; exclude _comprehension.txt sidecars
    all_files = sorted(
        f for f in p.iterdir()
        if f.suffix.lower() in (".cha", ".txt")
        and "_comprehension" not in f.stem
    )
    if not all_files:
        print(f"No .cha or .txt transcript files found in '{transcripts_dir}'.")
        return

    # Apply story filter
    if story_filter:
        all_files = [f for f in all_files if story_filter in f.name.lower()]
        if not all_files:
            print(f"No files matching story '{story_filter}' found.")
            return

    print(f"Found {len(all_files)} transcript(s)"
          + (f" [filtered: {story_filter}]" if story_filter else "") + ".")

    tokenizer = model = torch_mod = None
    if use_bert:
        tokenizer, model, torch_mod = load_gabert()
        if tokenizer is None:
            print("WARNING: GABert failed to load. Running rules-only.")
            use_bert = False

    all_rows: list = []

    for fpath in all_files:
        tid   = fpath.stem
        story = infer_story(fpath.name)
        if story is None:
            print(f"SKIP: cannot infer story from '{fpath.name}' "
                  f"(filename should include: birds/goats/dog/cat)")
            continue

        # ── Load transcript (CHAT or plain text) ─────────────────────────────
        try:
            narrative, comp_responses, detected_speaker = load_transcript(
                str(fpath), child_speaker)
        except Exception as e:
            print(f"ERROR loading '{fpath.name}': {e}")
            continue

        if not narrative.strip():
            print(f"WARNING: No child speech extracted from '{fpath.name}'. "
                  f"Check speaker code (detected: {detected_speaker}).")
            continue

        if fpath.suffix.lower() == ".cha":
            print(f"  {fpath.name}: CHAT file, child speaker={detected_speaker}, "
                  f"{len(narrative.split())} words extracted.")
        elif detected_speaker != "n/a":
            print(f"  {fpath.name}: CHAT-style .txt, child speaker={detected_speaker}, "
                  f"{len(narrative.split())} words extracted (examiner turns excluded).")
        else:
            print(f"  {fpath.name}: plain text.")

        # For .txt files, check for a sidecar _comprehension.txt
        if fpath.suffix.lower() == ".txt" and not comp_responses:
            comp_path = p / f"{tid}_comprehension.txt"
            comp_responses = parse_comprehension_file(str(comp_path))

        # ── Score all sections ────────────────────────────────────────────────
        sec_a = score_section_A(narrative, story)
        sec_b = score_section_B(sec_a)
        sec_c = score_section_C(narrative)
        sec_d = score_section_D(comp_responses, story) if comp_responses else None

        sec_a_bert = None
        if use_bert:
            sec_a_bert = score_section_A_gabert(
                narrative, story, tokenizer, model, torch_mod)

        print_full_report(tid, story, sec_a, sec_b, sec_c, sec_d, sec_a_bert)
        all_rows.append(flatten_for_csv(tid, story, "rules",
                                        sec_a, sec_b, sec_c, sec_d))
        if sec_a_bert:
            sec_b_bert = score_section_B(sec_a_bert)
            all_rows.append(flatten_for_csv(tid, story, "bert",
                                            sec_a_bert, sec_b_bert, sec_c, sec_d))

    if all_rows:
        export_csv(all_rows, output_csv)


# =============================================================================
# SECTION 10: ENTRY POINT & DEMO
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="MAIN Irish (Gaeilge) Scorer — Sections A, B, C, D"
    )
    parser.add_argument("--transcripts", default="transcripts",
                        help="Folder containing .cha or .txt transcript files")
    parser.add_argument("--bert", action="store_true",
                        help="Also score with GABert (requires: pip install transformers torch)")
    parser.add_argument("--output", default="main_scores.csv",
                        help="Output CSV filename")
    parser.add_argument("--story", default=None,
                        help="Only process files for this story (dog/birds/cat/goats)")
    parser.add_argument("--child-speaker", default=None, dest="child_speaker",
                        help="Override child speaker code (e.g. SP02). "
                             "If omitted, auto-detected from @ID tier in .cha files.")
    parser.add_argument("--demo", action="store_true",
                        help="Run built-in Baby Birds demo")
    parser.add_argument("--demo-chat", action="store_true",
                        help="Run CHAT parser demo on the example dog transcript")
    args = parser.parse_args()

    if args.demo_chat:
        # ── Inline CHAT demo using the dog transcript from the paper ──────────
        sample_cha = """\
@Begin
@Languages:\tgle
@Participants:\tSP02 21MNM4SI33 Target_Child, SP01 Marion Investigator
@ID:\tgle|change_corpus_later|SP02|14;09.12|male|||Target_Child|||
@ID:\tgle|change_corpus_later|SP01|||||Investigator|||
@Media:\t21MNM4SI33(Irish_Dog), audio
*SP01:\tanois ba mhaith liom go n-inseodh tusa an scéal .
*SP01:\tso féach ar na pictiúir agus déan iarracht an scéal is fearr ar
\tféidir leatsa a insint .
*SP02:\tso bhí an madra &em a &+s spraoi taobh amuigh agus chonaic sé an luch
\tsuí in aice leis an crann .
*SP01:\t&hm [/] .
*SP02:\tcause[?] <bhí sé> [///] bhí an madra ag iarraidh <ag
\tfháil> [?] greim air agus rith an luch uaidh[=? away] .
*SP01:\t&hm [/] .
*SP02:\tagus &em bhuail sé fhéin mar bhí sé gortaithe .
*SP01:\t&oh sea ach céard eile a tharla sna pictiúirí seo ?
*SP02:\tbhuail an madra é fhéin in aghaidh an <chrann nuair>
\t[?] bhí duine éicint ag tíocht .
*SP02:\tle siopadóireacht agus bhí &+bal balún buí ag an lámh .
*SP01:\t&hm [/] .
*SP02:\tagus &em (.) &em [/] chonaic sé an madra is &em scanraigh sé agus
\tlig sé leis an balún .
%com:\tsounds like 'scantraigh sé'
*SP02:\tag an balún san crann .
*SP01:\t&hm .
*SP02:\tbhí an [/] leaid &em ag tarraingt an balún as an crann .
*SP02:\tagus bhí an madra (a)g iarraidh ispín agus then .
*SP02:\tchuaigh an buachaill an balún as agus bhí an madra ithe ispín .
*SP01:\tar thaithnigh an scéal leat ?
*SP02:\tyeah .
*SP01:\tcén fáth a léimeann an madra amach chun tosaigh ?
*SP02:\tmar bhí sé ag iarraidh an luch .
*SP01:\tagus sa phictiúr seo . cén chaoi a mbraitheann an madra ?
*SP02:\t&em bhuail sé fhéin in aghaidh an [/] crann ?
*SP01:\tsea agus cén chaoi a n-airíonn sé ?
*SP02:\t&em gortaithe .
*SP01:\tagus cén fáth a bhfuil an madra gortaithe ?
*SP02:\tmar bhuail sé fhéin in aghaidh an cloigeann san crann .
*SP01:\ttuigim agus sa phictiúr seo . cén fáth a léimeann an buachaill suas ?
*SP02:\tmar xxx sé ag iarraidh an balún ar ais .
*SP01:\tagus sa phicitiúr seo . cén chaoi a mothaíonn an buachaill ?
*SP02:\t<bhí sé ag baint> [//] bhí sé &em a bhaint dhó leis an crann .
*SP01:\ttuigim agus cén chaoi a mothaíonn nó cén chaoi a n-airíonn an buachaill ?
*SP02:\tsásta .
*SP01:\tsásta agus cén fáth a n-airíonn sé sásta ?
*SP02:\tmar fuair sé an balún ar ais .
*SP01:\ttuigim agus sa gceann seo arís . cén fáth a sciobann an madra na hispíní ?
*SP02:\tbhuel bhí ocras air .
*SP01:\toh tuigim agus ar ais go dtí an ceann seo arís .
*SP01:\tlig ort fhéin go bhfeiceann an buachaill an madra . cén chaoi a n-airíonn an buachaill ?
*SP02:\t&em [/] crosta b'fhéidir .
*SP01:\tagus meas tú cén fáth a mbeadh sé crosta ?
*SP02:\tmar d'ith sé (a)n ispíní .
*SP01:\ttuigim agus an cheist dheireanach .
*SP01:\tan mbeidh an buachaill cairdiúil leis an madra an gceapann tú ?
*SP02:\t&em no .
*SP01:\tagus cén fáth ?
*SP02:\tmar a d'ith sé an[=? na] ispíní arís .
@End
"""
        import tempfile, os
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cha",
                                         encoding="utf-8", delete=False) as tf:
            tf.write(sample_cha)
            tmp_path = tf.name

        print("\n=== DEMO: CHAT parser on dog transcript ===\n")
        narrative, comp_responses, speaker = load_transcript(tmp_path)
        os.unlink(tmp_path)

        print(f"Detected child speaker : {speaker}")
        print(f"\nExtracted narrative ({len(narrative.split())} words):")
        print(f"  {narrative}")
        print(f"\nExtracted comprehension responses ({len(comp_responses)} questions):")
        for q, r in comp_responses.items():
            print(f"  {q}: {r}")

        print("\n--- Scoring extracted narrative ---")
        sec_a = score_section_A(narrative, "dog")
        sec_b = score_section_B(sec_a)
        sec_c = score_section_C(narrative)
        sec_d = score_section_D(comp_responses, "dog") if comp_responses else None
        print_full_report("demo_chat_dog", "dog", sec_a, sec_b, sec_c, sec_d)

    elif args.demo:
        # Constructed narrative covering all 16 rubric components (A1-A16)
        narrative = (
            # A1 Setting: time + place
            "Lá amháin bhí Mamaí Éan agus a cuid éiníní beaga ina gcuid nead sa chrann. "
            # A2 IST/IE: chicks hungry
            "Bhí ocras ar na héiníní beaga agus bhí siad ag caoineadh le haghaidh bia. "
            # A3 Goal: mother wants food
            "Bhí Mamaí Éan ag iarraidh bia a fháil do na héiníní. "
            # A4 Attempt: mother flies off
            "D'eitil an Mhamaí Éan léi ag cuardach bia. "
            # A5 Outcome: returns with worm
            "Tháinig an Mhamaí Éan ar ais le péist do na héiníní. "
            # A6 Reaction: joy
            "Bhí áthas ar na héiníní beaga agus ní raibh ocras orthu níos mó. "
            # A7 IST/IE: cat notices chicks alone
            "Chonaic an cat go raibh na héiníní beaga ina n-aonar sa nead. "
            # A8 Goal: cat wants to eat chicks
            "Bhí an cat ag iarraidh na héiníní beaga a ithe. "
            # A9 Attempt: cat climbs tree
            "Chuaigh an cat suas sa gcrann ag dreapadóireacht go mall. "
            # A10 Outcome: nearly catches chick
            "Ba bheag nár rug an cat ar cheann de na héiníní. "
            # A11 Reaction: cat happy; chicks scared
            "Bhí an cat sásta ach bhí eagla mhór ar na héiníní agus bhí siad ag caoineadh. "
            # A12 IST/IE: dog notices danger
            "Chonaic an madra go raibh an t-éan i gcontúirt agus go raibh greim ag an gcat air. "
            # A13 Goal: dog decides to save chicks
            "Shocraigh an madra na héiníní a shábháil agus an cat a stopadh. "
            # A14 Attempt: dog grabs tail
            "Rug an madra greim ar eireaball an chait agus tharraing sé é. "
            # A15 Outcome: cat flees; chick saved
            "Chuir an madra ruaig ar an gcat agus sábháladh na héiníní. "
            # A16 Reaction: dog proud; cat sulky; chicks/mother relieved
            "Bhí an madra bródúil. Bhí an cat crosta agus bhí díomá air. "
            "Bhí faoiseamh ar na héiníní agus bhí an Mhamaí Éan sásta."
        )
        comp_responses = {
            "D1":  "Ag iarraidh bia a fháil do na héiníní.",
            "D2":  "Tá ocras orthu.",
            "D4":  "Ag iarraidh an t-éan a ithe.",
            "D5":  "Feargach — ní bhfuair sé na héiníní.",
            "D7":  "Ag iarraidh na héiníní a shábháil.",
            "D8":  "Bhí sé sásta — shábháil sé na héiníní.",
            "D10": "An madra — shábháil sé na héiníní.",
        }
        print("\n=== DEMO: Baby Birds (Eanlaith) ===")
        sec_a = score_section_A(narrative, "birds")
        sec_b = score_section_B(sec_a)
        sec_c = score_section_C(narrative)
        sec_d = score_section_D(comp_responses, "birds")
        print_full_report("demo_birds", "birds", sec_a, sec_b, sec_c, sec_d)

        if args.bert:
            tok, mdl, tch = load_gabert()
            if tok:
                sec_a_bert = score_section_A_gabert(narrative, "birds", tok, mdl, tch)
                print("\n=== DEMO with GABert ===")
                print_full_report("demo_birds_bert", "birds",
                                  sec_a, sec_b, sec_c, sec_d, sec_a_bert)
    else:
        process_transcripts(
            transcripts_dir=args.transcripts,
            use_bert=args.bert,
            output_csv=args.output,
            child_speaker=args.child_speaker,
            story_filter=args.story,
        )
        # Constructed narrative covering all 16 rubric components (A1-A16)
        narrative = (
            # A1 Setting: time + place
            "Lá amháin bhí Mamaí Éan agus a cuid éiníní beaga ina gcuid nead sa chrann. "
            # A2 IST/IE: chicks hungry
            "Bhí ocras ar na héiníní beaga agus bhí siad ag caoineadh le haghaidh bia. "
            # A3 Goal: mother wants food
            "Bhí Mamaí Éan ag iarraidh bia a fháil do na héiníní. "
            # A4 Attempt: mother flies off
            "D'eitil an Mhamaí Éan léi ag cuardach bia. "
            # A5 Outcome: returns with worm
            "Tháinig an Mhamaí Éan ar ais le péist do na héiníní. "
            # A6 Reaction: joy
            "Bhí áthas ar na héiníní beaga agus ní raibh ocras orthu níos mó. "
            # A7 IST/IE: cat notices chicks alone
            "Chonaic an cat go raibh na héiníní beaga ina n-aonar sa nead. "
            # A8 Goal: cat wants to eat chicks
            "Bhí an cat ag iarraidh na héiníní beaga a ithe. "
            # A9 Attempt: cat climbs tree
            "Chuaigh an cat suas sa gcrann ag dreapadóireacht go mall. "
            # A10 Outcome: nearly catches chick
            "Ba bheag nár rug an cat ar cheann de na héiníní. "
            # A11 Reaction: cat happy; chicks scared
            "Bhí an cat sásta ach bhí eagla mhór ar na héiníní agus bhí siad ag caoineadh. "
            # A12 IST/IE: dog notices danger
            "Chonaic an madra go raibh an t-éan i gcontúirt agus go raibh greim ag an gcat air. "
            # A13 Goal: dog decides to save chicks
            "Shocraigh an madra na héiníní a shábháil agus an cat a stopadh. "
            # A14 Attempt: dog grabs tail
            "Rug an madra greim ar eireaball an chait agus tharraing sé é. "
            # A15 Outcome: cat flees; chick saved
            "Chuir an madra ruaig ar an gcat agus sábháladh na héiníní. "
            # A16 Reaction: dog proud; cat sulky; chicks/mother relieved
            "Bhí an madra bródúil. Bhí an cat crosta agus bhí díomá air. "
            "Bhí faoiseamh ar na héiníní agus bhí an Mhamaí Éan sásta."
        )

        # Sample comprehension responses illustrating all D scoring paths
        comp_responses = {
            "D1":  "Ag iarraidh bia a fháil do na héiníní.",       # correct
            "D2":  "Tá ocras orthu.",                               # correct + has_explanation
            # D3 auto-scored as 1 because D2 has explanation
            "D4":  "Ag iarraidh an t-éan a ithe.",                  # correct
            "D5":  "Feargach — ní bhfuair sé na héiníní.",          # correct + has_explanation
            # D6 auto-scored as 1
            "D7":  "Ag iarraidh na héiníní a shábháil.",            # correct
            "D8":  "Bhí sé sásta — shábháil sé na héiníní.",        # correct + has_explanation
            # D9 auto-scored as 1
            "D10": "An madra — shábháil sé na héiníní.",            # correct
        }

        print("\n=== DEMO: Baby Birds (Eanlaith) ===")
        sec_a = score_section_A(narrative, "birds")
        sec_b = score_section_B(sec_a)
        sec_c = score_section_C(narrative)
        sec_d = score_section_D(comp_responses, "birds")
        print_full_report("demo_birds", "birds", sec_a, sec_b, sec_c, sec_d)

        if args.bert:
            tok, mdl, tch = load_gabert()
            if tok:
                sec_a_bert = score_section_A_gabert(
                    narrative, "birds", tok, mdl, tch)
                print("\n=== DEMO with GABert ===")
                print_full_report("demo_birds_bert", "birds",
                                  sec_a, sec_b, sec_c, sec_d, sec_a_bert)
