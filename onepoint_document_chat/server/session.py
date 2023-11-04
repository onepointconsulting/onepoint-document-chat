from typing import List, Dict


class Session:
    """Session object."""

    def __init__(
        self,
        # Id of the session
        id: str,
    ):
        self.id = id
        self.messages_history: List[str] = []
        sessions_id[id] = self

    def delete(self):
        """Delete the session."""
        sessions_id.pop(self.id, None)

    def messages_history_str(self) -> str:
        "\n\n".join(self.messages_history)


def add_message(sid: str, data: str) -> Session:
    if sid in sessions_id:
        sessions_id[sid].messages_history.append(data)
    else:
        Session(sid).messages_history.append(data)
    return sessions_id[sid]


def delete_session(sid: str):
    if sid in sessions_id:
        sessions_id[sid].delete()


sessions_id: Dict[str, Session] = {}
