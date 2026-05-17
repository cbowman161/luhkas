from models import get_model


class ChatAgent:
    def __init__(self):
        self.model = get_model("chat")

    def answer(self, user_input):
        prompt = f"""
You are LUHKAS Brain, a helpful local assistant.

User:
{user_input}

Assistant:
"""
        return self.model.generate(prompt, think=False)