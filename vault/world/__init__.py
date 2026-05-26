"""World knowledge vault — offline Wikipedia + media corpus.

Sibling to MemoryStore but identity-free. Holds reference data the assistant
consults when the user asks about world facts, not personal facts.
"""
from world.world_store import WorldKnowledgeStore

__all__ = ["WorldKnowledgeStore"]
