"""Shared constants for the PanoptOS training pipeline."""

import string

SKILLS = [
    "overall",
    "agility",
    "attack",
    "construction",
    "cooking",
    "crafting",
    "defence",
    "farming",
    "firemaking",
    "fishing",
    "fletching",
    "herblore",
    "hitpoints",
    "hunter",
    "magic",
    "mining",
    "prayer",
    "ranged",
    "runecraft",
    "sailing",
    "slayer",
    "smithing",
    "strength",
    "thieving",
    "woodcutting",
]

ACTIVITIES = [
    "abyssal_sire",
    "alchemical_hydra",
    "amoxliatl",
    "araxxor",
    "artio",
    "barrows_chests",
    "bounty_hunter_hunter",
    "bounty_hunter_legacy_hunter",
    "bounty_hunter_legacy_rogue",
    "bounty_hunter_rogue",
    "bryophyta",
    "callisto",
    "calvar_ion",
    "cerberus",
    "chambers_of_xeric",
    "chambers_of_xeric_challenge_mode",
    "chaos_elemental",
    "chaos_fanatic",
    "clue_scrolls_all",
    "clue_scrolls_beginner",
    "clue_scrolls_easy",
    "clue_scrolls_elite",
    "clue_scrolls_hard",
    "clue_scrolls_master",
    "clue_scrolls_medium",
    "collections_logged",
    "colosseum_glory",
    "commander_zilyana",
    "corporeal_beast",
    "crazy_archaeologist",
    "dagannoth_prime",
    "dagannoth_rex",
    "dagannoth_supreme",
    "deranged_archaeologist",
    "doom_of_mokhaiotl",
    "duke_sucellus",
    "general_graardor",
    "giant_mole",
    "grotesque_guardians",
    "hespori",
    "k_ril_tsutsaroth",
    "kalphite_queen",
    "king_black_dragon",
    "kraken",
    "kree_arra",
    "lms_rank",
    "lunar_chests",
    "mimic",
    "nex",
    "nightmare",
    "obor",
    "phantom_muspah",
    "phosani_s_nightmare",
    "pvp_arena_rank",
    "rifts_closed",
    "sarachnis",
    "scorpia",
    "scurrius",
    "shellbane_gryphon",
    "skotizo",
    "sol_heredit",
    "soul_wars_zeal",
    "spindel",
    "tempoross",
    "the_corrupted_gauntlet",
    "the_gauntlet",
    "the_hueycoatl",
    "the_leviathan",
    "the_royal_titans",
    "the_whisperer",
    "theatre_of_blood",
    "theatre_of_blood_hard_mode",
    "thermonuclear_smoke_devil",
    "tombs_of_amascut",
    "tombs_of_amascut_expert_mode",
    "tzkal_zuk",
    "tztok_jad",
    "vardorvis",
    "venenatis",
    "vet_ion",
    "vorkath",
    "wintertodt",
    "yama",
    "zalcano",
    "zulrah",
]

COMBAT_SKILLS = [
    "attack",
    "strength",
    "defence",
    "ranged",
    "magic",
    "hitpoints",
    "prayer",
]
GATHERING_SKILLS = ["mining", "fishing", "woodcutting", "hunter", "farming"]
NON_KC_ACTIVITIES = [
    "bounty_hunter_hunter",
    "bounty_hunter_legacy_hunter",
    "bounty_hunter_legacy_rogue",
    "bounty_hunter_rogue",
    "collections_logged",
    "colosseum_glory",
    "lms_rank",
    "pvp_arena_rank",
    "rifts_closed",
    "soul_wars_zeal",
]
KC_ACTIVITIES = [a for a in ACTIVITIES if a not in set(NON_KC_ACTIVITIES)]

RAW_FEATURES = SKILLS + ACTIVITIES
VELOCITY_FEATURES = [f"{skill}_velocity" for skill in SKILLS]
RATIO_FEATURES = [f"{skill}_ratio" for skill in SKILLS[1:]]
# Order must match the concatenation in features.PlayerSequence.calculate_features.
DERIVED_FEATURES = [
    # Cumulative skill distribution
    "combat_gathering_ratio",
    "skill_entropy",
    "max_skill_ratio",
    "n_skills",
    # Cumulative activity distribution
    "n_activities",
    "activity_entropy",
    "max_activity_ratio",
    # Per-interval skill dynamics
    "gain_entropy",
    "gain_cosine_similarity",
    "velocity_delta",
    "active_skills",
    # Per-interval activity dynamics
    "active_activities",
    "activity_switch",
    "activity_streak",
]

ALL_FEATURES = RAW_FEATURES + VELOCITY_FEATURES + RATIO_FEATURES + DERIVED_FEATURES

OVERALL_INDEX = ALL_FEATURES.index("overall")
SKILL_INDICES = [ALL_FEATURES.index(s) for s in SKILLS]
ACTIVITY_INDICES = [ALL_FEATURES.index(a) for a in ACTIVITIES]
KC_ACTIVITY_INDICES = [ALL_FEATURES.index(a) for a in KC_ACTIVITIES]
COMBAT_INDICES = [ALL_FEATURES.index(s) for s in COMBAT_SKILLS]
GATHERING_INDICES = [ALL_FEATURES.index(s) for s in GATHERING_SKILLS]

MIN_SEQUENCE_LENGTH = 3
MAX_SEQUENCE_LENGTH = 30
MIN_OVERALL_XP = 100_000

# Padding sentinels — must stay in sync across the collate, ONNX export, and
# the model's mask derivation.
# Features pad with -1: OSRS XP is nonnegative, so -1 is unambiguous.
# Names pad with 0: matches nn.Embedding(padding_idx=0).
FEATURE_PAD_VALUE = -1.0
NAME_PAD_VALUE = 0

ALL_CHARACTERS = string.ascii_lowercase + string.ascii_uppercase + string.digits + " _-"
STOI = {ch: i for i, ch in enumerate(ALL_CHARACTERS, start=1)}
ITOS = {i: ch for i, ch in enumerate(ALL_CHARACTERS, start=1)}


def encode(s: str) -> list[int]:
    return [STOI[c] for c in s]


def decode(seq: list[int]) -> str:
    return "".join(ITOS[i] for i in seq)
