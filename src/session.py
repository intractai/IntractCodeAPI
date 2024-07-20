from concurrent.futures import CancelledError
import threading
from src.users import SessionTracker

GLOBAL_MAIN_THREAD_ID = None

def thread_hook(username, *args):
    current_thread_id = threading.get_ident()

    session_tracker = SessionTracker.get_instance()
    activity_threads = session_tracker.get_user_activity_threads(username)

    if current_thread_id not in activity_threads:
        raise CancelledError("Cancelled by new request")


def set_main_thread_id():
    global GLOBAL_MAIN_THREAD_ID
    GLOBAL_MAIN_THREAD_ID = threading.get_ident()
