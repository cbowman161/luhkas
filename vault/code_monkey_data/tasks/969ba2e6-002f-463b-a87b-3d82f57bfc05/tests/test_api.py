import unittest
from src.api import schema, create_reminder, list_reminders, delete_reminder, check_status

class TestPublicApi(unittest.TestCase):
    def setUp(self):
        # Ensure we start with a clean state
        self.reminder_title = "test_reminder"
        # Clean up any existing test reminder
        delete_reminder(self.reminder_title)

    def test_schema_exists_and_is_dict(self):
        schema_result = schema()
        self.assertIsInstance(schema_result, dict)
        self.assertIn('package', schema_result)
        self.assertIn('endpoints', schema_result)
        self.assertIn('response_format', schema_result)
        self.assertIn('storage_notes', schema_result)

    def test_create_reminder_success(self):
        result = create_reminder("test_reminder", "2025-01-01 12:00:00", "Test message", "reminder")
        self.assertTrue(result['ok'])
        self.assertEqual(result['action'], 'create_reminder')
        self.assertEqual(result['message'], 'Reminder created successfully')
        self.assertIsNotNone(result['data'])
        self.assertIsNone(result['error'])

    def test_create_reminder_missing_fields(self):
        result = create_reminder("", "2025-01-01 12:00:00", "Test message", "reminder")
        self.assertFalse(result['ok'])
        self.assertEqual(result['action'], 'create_reminder')
        self.assertEqual(result['message'], 'All fields are required')
        self.assertIsNone(result['data'])
        self.assertIsNotNone(result['error'])

    def test_create_reminder_invalid_time_format(self):
        result = create_reminder("test_reminder", "invalid_time", "Test message", "reminder")
        self.assertFalse(result['ok'])
        self.assertEqual(result['action'], 'create_reminder')
        self.assertEqual(result['message'], 'Invalid time format')
        self.assertIsNone(result['data'])
        self.assertIsNotNone(result['error'])

    def test_create_reminder_duplicate_title(self):
        create_reminder("test_reminder", "2025-01-01 12:00:00", "Test message", "reminder")
        result = create_reminder("test_reminder", "2025-01-01 12:00:00", "Test message", "reminder")
        self.assertFalse(result['ok'])
        self.assertEqual(result['action'], 'create_reminder')
        self.assertEqual(result['message'], 'Reminder already exists')
        self.assertIsNone(result['data'])
        self.assertIsNotNone(result['error'])

    def test_list_reminders_empty(self):
        result = list_reminders()
        self.assertTrue(result['ok'])
        self.assertEqual(result['action'], 'list_reminders')
        self.assertEqual(result['message'], 'Reminders retrieved successfully')
        self.assertIsInstance(result['data'], list)
        self.assertEqual(len(result['data']), 0)

    def test_list_reminders_with_entries(self):
        create_reminder("test_reminder1", "2025-01-01 12:00:00", "Test message 1", "reminder")
        create_reminder("test_reminder2", "2025-01-01 13:00:00", "Test message 2", "reminder")
        result = list_reminders()
        self.assertTrue(result['ok'])
        self.assertEqual(result['action'], 'list_reminders')
        self.assertEqual(result['message'], 'Reminders retrieved successfully')
        self.assertIsInstance(result['data'], list)
        self.assertEqual(len(result['data']), 2)

    def test_check_status_exists(self):
        create_reminder("test_reminder", "2025-01-01 12:00:00", "Test message", "reminder")
        result = check_status("test_reminder")
        self.assertTrue(result['ok'])
        self.assertEqual(result['action'], 'check_status')
        self.assertEqual(result['message'], 'Reminder found')
        self.assertIsNotNone(result['data'])
        self.assertIsNone(result['error'])

    def test_check_status_not_exists(self):
        result = check_status("nonexistent_reminder")
        self.assertFalse(result['ok'])
        self.assertEqual(result['action'], 'check_status')
        self.assertEqual(result['message'], 'Reminder not found')
        self.assertIsNone(result['data'])
        self.assertIsNotNone(result['error'])

    def test_delete_reminder_success(self):
        create_reminder("test_reminder", "2025-01-01 12:00:00", "Test message", "reminder")
        result = delete_reminder("test_reminder")
        self.assertTrue(result['ok'])
        self.assertEqual(result['action'], 'delete_reminder')
        self.assertEqual(result['message'], 'Reminder deleted successfully')
        self.assertIsNone(result['data'])
        self.assertIsNone(result['error'])

    def test_delete_reminder_not_exists(self):
        result = delete_reminder("nonexistent_reminder")
        self.assertFalse(result['ok'])
        self.assertEqual(result['action'], 'delete_reminder')
        self.assertEqual(result['message'], 'Reminder not found')
        self.assertIsNone(result['data'])
        self.assertIsNotNone(result['error'])

    def test_full_lifecycle(self):
        # Create
        create_result = create_reminder("lifecycle_test", "2025-01-01 12:00:00", "Test message", "reminder")
        self.assertTrue(create_result['ok'])

        # List and verify
        list_result = list_reminders()
        self.assertTrue(list_result['ok'])
        self.assertEqual(len(list_result['data']), 1)
        self.assertEqual(list_result['data'][0]['title'], "lifecycle_test")

        # Check status
        check_result = check_status("lifecycle_test")
        self.assertTrue(check_result['ok'])

        # Delete
        delete_result = delete_reminder("lifecycle_test")
        self.assertTrue(delete_result['ok'])

        # List again and verify it's gone
        list_result = list_reminders()
        self.assertTrue(list_result['ok'])
        self.assertEqual(len(list_result['data']), 0)

if __name__ == '__main__':
    unittest.main()
