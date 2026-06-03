from pathlib import Path
import os


ROOT_DIR = Path(__file__).resolve().parent
DB_PATH = str(ROOT_DIR / "vault_v2.db")
CAPABILITIES_DIR = ROOT_DIR / "capabilities"
SKILLS_DIR = ROOT_DIR / "skills"
LOGS_DIR = ROOT_DIR / "logs"
DATA_DIR = ROOT_DIR / "data"
PEOPLE_DIR = DATA_DIR / "people"
FACE_REFERENCES_DIR = DATA_DIR / "face_references"
CLASSROOM_DIR = DATA_DIR / "classroom"

INSTALLED_CAPABILITIES_DIR = ROOT_DIR / "installed_capabilities"

SYSTEM_CAPABILITIES_PATH = str(CAPABILITIES_DIR / "system.json")
LEARNED_CAPABILITIES_PATH = str(CAPABILITIES_DIR / "learned.json")
SKILLS_REGISTRY_PATH = str(SKILLS_DIR / "skills.json")
TASK_LOG_DIR = str(LOGS_DIR / "tasks")

CODE_MONKEY_URL = "http://127.0.0.1:8765"
SCOUT_URL = os.environ.get("VAULT_SCOUT_URL", "http://luhkas-scout:5000")
SCOUT_ROBOT_URL = os.environ.get("VAULT_SCOUT_ROBOT_URL", "http://luhkas-scout:5001")
SCOUT_BATTERY_URL = os.environ.get("VAULT_SCOUT_BATTERY_URL", "http://luhkas-scout:5003")
OLLAMA_URL = os.environ.get("VAULT_OLLAMA_URL", "http://localhost:11434")

VAULT_ROUTER_MODEL = os.environ.get("VAULT_ROUTER_MODEL", "qwen2.5:3b-instruct")
VAULT_CHAT_MODEL = os.environ.get("VAULT_CHAT_MODEL", "qwen3:8b")
VAULT_REASONER_MODEL = os.environ.get("VAULT_REASONER_MODEL", "qwen3:30b")
# Dedicated tutor model for classroom mode. The classroom controller
# evicts router/chat/vision/coder before loading this so the full
# 24 GB VRAM is available to it. qwen3:30b (~17.3 GB Q4_K_M) is the
# default because it's already pulled (it's also the reasoner default)
# and fits comfortably alongside bge-m3 (~1.3 GB) + KV cache. Override
# with VAULT_TEACHER_MODEL=qwen2.5:32b-instruct (needs ~20 GB pull) if
# you want a different instructor profile.
VAULT_TEACHER_MODEL = os.environ.get("VAULT_TEACHER_MODEL", "qwen3:30b")
VAULT_PLANNER_MODEL = os.environ.get("VAULT_PLANNER_MODEL", VAULT_REASONER_MODEL)
VAULT_ANALYST_MODEL = os.environ.get("VAULT_ANALYST_MODEL", VAULT_REASONER_MODEL)
VAULT_CODER_MODEL = os.environ.get("VAULT_CODER_MODEL", "qwen3-coder:30b")
VAULT_FAST_CODER_MODEL = os.environ.get("VAULT_FAST_CODER_MODEL", "qwen2.5-coder:14b")
VAULT_VISION_MODEL = os.environ.get("VAULT_VISION_MODEL", "qwen2.5vl:7b")
VAULT_EMBED_MODEL = os.environ.get("VAULT_EMBED_MODEL", "bge-m3")
# IMMEDIATE = router/chat/vision/embed: models the user-facing loop
# touches all the time. Kept resident long enough that an interactive
# session never pays a cold-load (~5-7s per model). Was 24h, which
# meant the warm models never evicted — that combined with the
# ingest's native bge-m3 (separate copy, ~3 GB) and Ollama's own
# bge-m3 (~1.3 GB) pushed VRAM use to ~18/24 GB on the RTX 3090,
# starving the ingest embedder of headroom and dropping ingest
# throughput from 19-21/s to 16-17/s.
#
# 30m is the sweet spot:
#  - During active use, the supervisor pre-warms after every pause,
#    so models stay refreshed throughout an interactive session.
#  - During long idle (e.g., overnight), models evict naturally and
#    VRAM frees up for the ingest embedder → peak throughput again.
#  - First turn after a long idle period takes one cold load (~5s);
#    every subsequent turn is snappy.
VAULT_IMMEDIATE_KEEP_ALIVE = os.environ.get("VAULT_IMMEDIATE_KEEP_ALIVE", "30m")
VAULT_BACKGROUND_KEEP_ALIVE = os.environ.get("VAULT_BACKGROUND_KEEP_ALIVE", "5m")
# Roles to pre-warm on vault startup and on ingest-pause.
#
# router + chat are cheap to keep resident (~8 GB combined) and the
# ingest throughput cost is small (~25/s cold vs ~23.5/s with these
# warm = 5% gap). They're touched on every user turn, so keeping them
# warm makes the first turn of any chat session feel instant instead
# of paying ~5-7s of cold-load per model.
#
# vision is intentionally EXCLUDED: qwen2.5vl:7b loads ~13 GB of VRAM
# (model + vision encoder + KV cache). Keeping it warm dropped ingest
# from ~25/s to ~17/s — a 40% penalty for a feature called maybe a
# handful of times per session. Lets it cold-load on demand; first
# "what do you see" pays a 5-7s wait but subsequent vision queries in
# the same 30m keep_alive window stay snappy.
VAULT_WARM_MODEL_ROLES = os.environ.get("VAULT_WARM_MODEL_ROLES", "router,chat")

OLLAMA_VISION_MODEL = VAULT_VISION_MODEL
