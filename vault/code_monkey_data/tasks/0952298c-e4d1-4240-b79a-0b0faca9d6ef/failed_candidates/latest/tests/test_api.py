import unittest
from src.api import schema, add, get, remove

class TestPublicApi(unittest.TestCase):
    def setUp(self):
        self.s = schema()
        self.endpoints = self.s["endpoints"]

    def test_schema_contract(self):
        self.assertIsInstance(self.s, dict)
        self.assertIn("endpoints", self.s)

    def _assert_response_envelope(self, response):
        self.assertSetEqual(set(response.keys()), {"ok", "action", "message", "data", "error"})

    def test_public_api_lifecycle(self):
        for endpoint in self.endpoints:
            if endpoint == 'add':
                response = add('test', 'test')
                self._assert_response_envelope(response)
                self.assertEqual(response['ok'], True)
                self.assertEqual(response['action'], 'add')
                self.assertEqual(response['data']['key'], 'test')
                self.assertEqual(response['data']['value'], 'test')
            elif endpoint == 'get':
                add('test', 'test')
                response = get('test')
                self._assert_response_envelope(response)
                self.assertEqual(response['ok'], True)
                self.assertEqual(response['action'], 'get')
                self.assertEqual(response['data']['key'], 'test')
                self.assertEqual(response['data']['value'], 'test')
            elif endpoint == 'remove':
                add('test', 'test')
                response = remove('test')
                self._assert_response_envelope(response)
                self.assertEqual(response['ok'], True)
                self.assertEqual(response['action'], 'remove')
                self.assertEqual(response['data']['key'], 'test')
                self.assertEqual(response['data']['value'], 'test')

if __name__ == "__main__":
    unittest.main()
