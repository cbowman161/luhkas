from storage.state_store import StateStore
from storage.vector_store import VectorStore


class Blackboard:
    def __init__(self):
        self.state = StateStore()
        self.memory = VectorStore()
        self.session = {
            "pending_decision": None,
            "last_skill": None,
        }

    # ----------------------------
    # SESSION STATE
    # ----------------------------

    def reset_session(self):
        self.session = {
            "pending_decision": None,
            "last_skill": self.session.get("last_skill"),
        }

    def set(self, key, value):
        self.session[key] = value

    def get_session_value(self, key, default=None):
        return self.session.get(key, default)

    def set_pending_decision(self, decision):
        self.session["pending_decision"] = decision

    def get_pending_decision(self):
        return self.session.get("pending_decision")

    def clear_pending_decision(self):
        self.session["pending_decision"] = None

    # ----------------------------
    # TASKS
    # ----------------------------

    def init_task(self, task_id, goal):
        self.state.create_task(task_id, goal)

    def update(self, task_id, data):
        self.state.add_history(task_id, data)

    def set_result(self, task_id, result):
        self.state.set_task_result(task_id, result)

    def get(self, task_id):
        task = self.state.get_task(task_id)

        if not task:
            return None

        history = self.state.get_history(task_id)

        return {
            **task,
            "history": history,
        }

    def list_tasks(self):
        return self.state.list_tasks()

    # ----------------------------
    # MEMORY
    # ----------------------------

    def remember(self, content, metadata=None):
        self.memory.add(content, metadata)

    def recall(self, query):
        return self.memory.search(query)