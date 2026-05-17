class VectorStore:
    def __init__(self):
        self.memory = []

    def add(self, content, metadata=None):
        self.memory.append({
            "content": content,
            "metadata": metadata or {}
        })

    def search(self, query, top_k=5):
        # 🔥 placeholder (replace with embeddings later)
        return self.memory[-top_k:]
