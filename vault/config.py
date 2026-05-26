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
VAULT_PLANNER_MODEL = os.environ.get("VAULT_PLANNER_MODEL", VAULT_REASONER_MODEL)
VAULT_ANALYST_MODEL = os.environ.get("VAULT_ANALYST_MODEL", VAULT_REASONER_MODEL)
VAULT_CODER_MODEL = os.environ.get("VAULT_CODER_MODEL", "qwen3-coder:30b")
VAULT_FAST_CODER_MODEL = os.environ.get("VAULT_FAST_CODER_MODEL", "qwen2.5-coder:14b")
VAULT_VISION_MODEL = os.environ.get("VAULT_VISION_MODEL", "qwen2.5vl:7b")
VAULT_EMBED_MODEL = os.environ.get("VAULT_EMBED_MODEL", "bge-m3")
VAULT_IMMEDIATE_KEEP_ALIVE = os.environ.get("VAULT_IMMEDIATE_KEEP_ALIVE", "24h")
VAULT_BACKGROUND_KEEP_ALIVE = os.environ.get("VAULT_BACKGROUND_KEEP_ALIVE", "5m")
VAULT_WARM_MODEL_ROLES = os.environ.get("VAULT_WARM_MODEL_ROLES", "router,chat,vision")

OLLAMA_VISION_MODEL = VAULT_VISION_MODEL
