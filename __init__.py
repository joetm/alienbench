import logging

# numexpr (imported by pandas) logs its thread-pool size at INFO on import.
logging.getLogger("numexpr").setLevel(logging.WARNING)
