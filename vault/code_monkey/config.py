import os
from pathlib import Path

DATA_DIR = Path('code_monkey_data')
TASKS_DIR = DATA_DIR / 'tasks'
DB_PATH = DATA_DIR / 'code_monkey.sqlite3'
DEFAULT_TIMEZONE = 'America/New_York'
OLLAMA_BASE_URL = os.environ.get('VAULT_OLLAMA_URL', 'http://localhost:11434').rstrip('/')
OLLAMA_GENERATE_URL = os.environ.get('BRAIN_OLLAMA_GENERATE_URL', OLLAMA_BASE_URL + '/api/generate')
BACKGROUND_KEEP_ALIVE = os.environ.get('VAULT_BACKGROUND_KEEP_ALIVE', '5m')
CODER_MODEL = os.environ.get('VAULT_CODER_MODEL', 'qwen3-coder:30b')
PLANNER_MODEL = os.environ.get('VAULT_PLANNER_MODEL', os.environ.get('VAULT_FAST_CODER_MODEL', 'qwen2.5-coder:14b'))
REPAIR_MODEL = os.environ.get('BRAIN_REPAIR_MODEL', CODER_MODEL)
