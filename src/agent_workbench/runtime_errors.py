class RuntimeCancelled(Exception):
    """Cooperative cancellation observed between graph nodes."""
    def __init__(self, state):
        super().__init__("run_cancelled")
        self.state = state
