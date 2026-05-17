from pathlib import Path
import os
import json
from typing import Dict, Any

def _cm_success(action, message, data=None):
    if message is None or message == '':
        message = str(action) + ' completed'
    return {'ok': True, 'action': str(action), 'message': str(message), 'data': data, 'error': None}

def _cm_failure(action, message, error, data=None):
    if message is None or message == '':
        message = str(action) + ' failed'
    if error is None or error == '':
        error = message
    return {'ok': False, 'action': str(action), 'message': str(message), 'data': data, 'error': str(error)}
SKILL_ROOT = Path(__file__).resolve().parent
DATA_DIR = SKILL_ROOT / 'data'
DATA_DIR.mkdir(parents=True, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

def add(key: str, value: Any):
    filename = f'{key}.json'
    path = DATA_DIR / filename
    try:
        with open(path, 'w', encoding='utf-8') as handle:
            json.dump({'value': value}, handle)
    except Exception as e:
        return _cm_failure('add', error=str(e))
    return _cm_success('add', data={'key': key, 'value': value})

def get(key: str):
    filename = f'{key}.json'
    path = DATA_DIR / filename
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return _cm_failure('get', f'No entry found for key {key}')
    except Exception as e:
        return _cm_failure('get', error=str(e))
    return _cm_success('get', data=data)

def remove(key: str):
    filename = f'{key}.json'
    path = DATA_DIR / filename
    try:
        os.remove(path)
    except FileNotFoundError:
        return _cm_failure('remove', f'No entry found for key {key}')
    except Exception as e:
        return _cm_failure('remove', error=str(e))
    return _cm_success('remove', data={'key': key})

def schema():
    return {'package': 'simple_storage', 'endpoints': {'add': {'args': ['key', 'value'], 'returns': 'dict'}, 'get': {'args': ['key'], 'returns': 'dict'}, 'remove': {'args': ['key'], 'returns': 'dict'}}, 'storage_notes': f"All data is stored in the '{DATA_DIR}' directory."}
