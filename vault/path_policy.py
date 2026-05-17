import os

from config import ROOT_DIR


class PathPolicy:
    def __init__(self, root=None):
        self.root = os.path.abspath(root or ROOT_DIR)
        self.skills_dir = os.path.join(self.root, "skills")
        self.capabilities_dir = os.path.join(self.root, "capabilities")
        self.learned_capabilities_path = os.path.join(
            self.capabilities_dir,
            "learned.json",
        )

    def normalize(self, path):
        return os.path.abspath(os.path.join(self.root, path))

    def is_inside(self, path, directory):
        path = os.path.abspath(path)
        directory = os.path.abspath(directory)
        return path == directory or path.startswith(directory + os.sep)

    def can_write_file(self, path):
        full = self.normalize(path)

        if self.is_inside(full, self.skills_dir):
            return {
                "allowed": True,
                "path": full,
                "reason": "Writes inside skills/ are allowed.",
            }

        if full == self.learned_capabilities_path:
            return {
                "allowed": True,
                "path": full,
                "reason": "Writing learned capabilities is allowed.",
            }

        return {
            "allowed": False,
            "path": full,
            "reason": "Writes are only allowed inside skills/ or capabilities/learned.json.",
        }

    def can_read_file(self, path):
        full = self.normalize(path)

        if self.is_inside(full, self.root):
            return {
                "allowed": True,
                "path": full,
                "reason": "Reads inside project root are allowed.",
            }

        return {
            "allowed": False,
            "path": full,
            "reason": "Reads outside project root are blocked.",
        }
