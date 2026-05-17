import json
import os
from pathlib import Path
from datetime import datetime

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
DATA_DIR.mkdir(parents=True, exist_ok=True)

def schema():
    return {'package': 'reminder_api', 'endpoints': {'create_reminder': {'args': ['title', 'time', 'message', 'type'], 'returns': 'dict'}, 'list_reminders': {'args': [], 'returns': 'dict'}, 'delete_reminder': {'args': ['title'], 'returns': 'dict'}, 'check_status': {'args': ['title'], 'returns': 'dict'}}, 'response_format': {'ok': 'bool', 'action': 'str', 'message': 'str', 'data': 'dict or list or None', 'error': 'str or None'}, 'storage_notes': 'Reminders stored as JSON files in src/data/ directory', 'notes': 'Each reminder file is named after the title and contains title, time, message, type fields'}

def create_reminder(title, time, message, type):
    if not title or not time or (not message) or (not type):
        return _cm_failure('create_reminder', 'All fields are required', 'Missing required fields')
    try:
        datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
    except ValueError:
        return _cm_failure('create_reminder', 'Invalid time format', 'Time must be in YYYY-MM-DD HH:MM:SS format')
    file_path = DATA_DIR / f'{title}.json'
    if file_path.exists():
        return _cm_failure('create_reminder', 'Reminder already exists', 'A reminder with this title already exists')
    reminder_data = {'title': title, 'time': time, 'message': message, 'type': type}
    try:
        with open(file_path, 'w') as handle:
            json.dump(reminder_data, handle)
    except Exception as e:
        return _cm_failure('create_reminder', 'Failed to save reminder', str(e))
    return _cm_success('create_reminder', 'Reminder created successfully', reminder_data)

def list_reminders():
    reminders = []
    try:
        for file_path in DATA_DIR.glob('*.json'):
            with open(file_path, 'r') as handle:
                data = json.load(handle)
                reminders.append(data)
    except Exception as e:
        return _cm_failure('list_reminders', 'Failed to read reminders', str(e))
    return _cm_success('list_reminders', 'Reminders retrieved successfully', reminders)

def delete_reminder(title):
    file_path = DATA_DIR / f'{title}.json'
    if not file_path.exists():
        return _cm_failure('delete_reminder', 'Reminder not found', 'No reminder with this title exists')
    try:
        os.remove(file_path)
    except Exception as e:
        return _cm_failure('delete_reminder', 'Failed to delete reminder', str(e))
    return _cm_success('delete_reminder', 'Reminder deleted successfully')

def check_status(title):
    file_path = DATA_DIR / f'{title}.json'
    if not file_path.exists():
        return _cm_failure('check_status', 'Reminder not found', 'No reminder with this title exists')
    try:
        with open(file_path, 'r') as handle:
            data = json.load(handle)
    except Exception as e:
        return _cm_failure('check_status', 'Failed to read reminder', str(e))
    return _cm_success('check_status', 'Reminder found', data)
