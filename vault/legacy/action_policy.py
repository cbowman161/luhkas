class ActionPolicy:
    def allowed_actions(self, history):
        if not history:
            return ["list_files", "write_file", "knowledge"]

        last = history[-1]
        action = last.get("action") or {}
        result = last.get("result") or {}

        action_type = action.get("type")
        status = result.get("status")

        if action_type in {"write_file", "patch_file"}:
            return ["command", "read_file", "patch_file", "knowledge"]

        if action_type == "read_file":
            return ["patch_file", "command", "knowledge"]

        if action_type == "list_files":
            return ["read_file", "write_file", "patch_file", "knowledge"]

        if action_type == "command":
            if status == "success":
                return ["knowledge", "read_file", "patch_file", "command"]
            return ["read_file", "patch_file", "write_file", "command"]

        if action_type == "knowledge":
            return ["write_file", "patch_file", "read_file", "command", "knowledge"]

        return ["write_file", "patch_file", "read_file", "command", "knowledge"]