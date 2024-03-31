from datetime import datetime, timedelta
import logging
import threading
import time
from typing import Optional, Union

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr

from src.database import make_db_connection, USER_TABLE


SECRET_KEY = '0GHa94H6pg89hlgvoJYeG+1LTKkjysoxYKIumfXirog='
ENCRYPTION_ALGORITHM = 'HS256'


logger = logging.getLogger(__name__)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl='token')
pwd_context = CryptContext(schemes=['bcrypt'], deprecated='auto')


### Session tracker ###

class SessionTracker:
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(
            cls: 'SessionTracker',
            model_provider = None,
            max_active_sessions: int = 100,
            eviction_interval: int = 300,
            max_session_idle_time: int = 600
        ) -> 'SessionTracker':
        """Get the singleton instance of the session tracker.

        Args:
            model_provider (ModelProvider): The model provider used to delete models when sessions are evicted.
            max_active_sessions (int, optional): The maximum number of users allowed. Defaults to 100.
            eviction_interval (int, optional): The interval to check for inactive users. Defaults to 300.
            max_session_idle_time (int, optional): The threshold to evict inactive users. Defaults to 600.

        Returns:
            SessionTracker: The singleton instance of the session tracker.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(model_provider, max_active_sessions, eviction_interval, max_session_idle_time)
        return cls._instance

    def __init__(self, model_provider, max_active_sessions: int, eviction_interval: int, max_session_idle_time: int):
        """Initialize the session tracker."""
        if SessionTracker._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            assert model_provider is not None, "Model provider must be provided!"
            SessionTracker._instance = self
            self.model_provider = model_provider
            self.max_active_sessions = max_active_sessions
            self.eviction_interval = eviction_interval
            self.max_session_idle_time = max_session_idle_time
            self.user_last_active = {}
            self._activity_threads = {} # Maps users to a dictionary of threads running for that user
            self._start_eviction_thread()

    def _start_eviction_thread(self):
        """Start the eviction thread to remove inactive users from the session tracker.
        
        The eviction thread runs every `eviction_interval` seconds and removes users who have not been active for
        `max_session_idle_time` seconds.
        """
        def eviction_process():
            while True:
                for user in list(self.user_last_active):
                    if not self.is_user_active(user):
                        self.evict_user(user)
                time.sleep(self.eviction_interval)

        thread = threading.Thread(target=eviction_process)
        thread.daemon = True
        thread.start()

    def is_user_active(self, username: str) -> bool:
        """Check if a user is active in the session tracker.
        
        Args:
            username (str): The username to check.
            
        Returns:
            bool: True if the user is active, False if the user is not active.
        """
        return self.user_last_active.get(username, 0) > time.time() - self.max_session_idle_time

    def evict_user(self, username: str) -> bool:
        """Evict a user from the session tracker and deletes their models.
        
        Args:
            username (str): The username to evict.
            
        Returns:
            bool: True if the user was evicted, False if the user was not evicted.
        """
        evicted = False
        with SessionTracker._lock:
            if username in self.user_last_active:
                del self.user_last_active[username]
                evicted = True

            if username in self._activity_threads:
                del self._activity_threads[username]
                evicted = True

        if evicted:
            self.model_provider.delete_model(username)
            logger.info(f"Evicted user {username}")

        return evicted

    def update_user_session(self, username: str, activity: Optional[str] = None) -> bool:
        """Update a user's session with the newest info.

        Sessions are always updated with the newest activity time, so that we can track
        which users are active and which are not.
        If an activity is provided, the thread for the running activity will be stored
        in the user's session data. This allows us to cancel the thread if the user
        starts a new activity. Note that terminated threads are not removed from the
        session data, but they will be overwritten if a new thread is started with the
        same activity name.
        
        Args:
            username (str): The username to update.
            activity (Optional[str], optional): The activity the user is performing. Defaults to None.
            
        Returns:
            bool: True if the user was updated, False if the user was not updated.
        """
        with SessionTracker._lock:
            # If the user is active or there is space for a new session
            if username in self.user_last_active or len(self.user_last_active) < self.max_active_sessions:
                # Update activity time
                self.user_last_active[username] = time.time()

                # Update activity thread if provided
                if activity is not None:
                    if username not in self._activity_threads:
                        self._activity_threads[username] = {}
                    self._activity_threads[username][activity] = threading.get_ident()

                return True
            else:
                return False
    
    def get_user_activity_threads(self, username: str) -> set:
        """Get the activity threads for a user.
        
        Args:
            username (str): The username to get the activity threads for.
            
        Returns:
            set: The activity threads for the user.
        """
        return set(self._activity_threads[username].values())


### Pydantic models ###

class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str


class UserLogin(BaseModel):
    username: str
    password: str


### User-based functions ###

async def authenticate_user(username: str, password: str) -> Union[dict, bool]:
    """Authenticate a user.
    
    Returns the user data if the username and password are correct, otherwise False.
    """
    user = await get_user_by_username(username)
    if not user or not pwd_context.verify(password, user[3]):
        return False
    return user


async def get_user_by_username(username: str) -> dict:
    """Get a user by their username.
    
    Args:
        username (str): The username of the user.
        
    Returns:
        dict: The user's information.
    """
    async with make_db_connection() as db:
        cursor = await db.execute(f"SELECT * FROM {USER_TABLE} WHERE username = ?", (username,))
        user = await cursor.fetchone()
        await cursor.close()
    return user


async def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    """Get a user's username by their token.

    Args:
        token (str): The token to get the username from.

    Returns:
        str: The user's username.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={'WWW-Authenticate': 'Bearer'},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ENCRYPTION_ALGORITHM])
        username: str = payload.get('sub')
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    return username


def validate_user_session(
        username: str = Depends(get_current_user),
        activity: Optional[str] = None,
    ) -> str:
    """Validate's a user's token and updates their session.
    
    This function should be a dependency for any route that requires a user to be logged in.

    Args:
        username (str): The username to validate.
        activity (Optional[str], optional): The activity the user is performing. Defaults to None.

    Returns:
        str: The username of the user.
    """
    sess_tracker = SessionTracker.get_instance()
    sess_tracker.update_user_session(username, activity)
    return username


async def get_user_by_token(token: str) -> dict:
    """Get a user's information by their token.

    Args:
        token (str): The token to get the user from.

    Returns:
        dict: The user's information.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={'WWW-Authenticate': 'Bearer'},
    )
    username = await get_current_user(token)
    user = await get_user_by_username(username)
    if user is None:
        raise credentials_exception
    return user


async def create_user(user: UserCreate):
    """Creates a new user by inserting their information into the database, ensuring the email is not already used."""
    async with make_db_connection() as db:
        # Prepare the query to check if the email or username already exists
        username_cursor = await db.execute(f'SELECT username FROM {USER_TABLE} WHERE username = ?', (user.username,))
        username_result = await username_cursor.fetchone()

        if username_result:
            # If a result is found, then the username already exists in the database.
            raise HTTPException(status_code=409, detail="Username already in use")

        email_cursor = await db.execute(f'SELECT email FROM {USER_TABLE} WHERE email = ?', (user.email,))
        email_result = await email_cursor.fetchone()

        if email_result:
            # If a result is found, then the email already exists in the database.
            raise HTTPException(status_code=409, detail="Email already in use")
        
        # Proceed with user creation if the email does not exist in the database
        hashed_password = pwd_context.hash(user.password)
        await db.execute('INSERT INTO {} (username, email, hashed_password) VALUES (?, ?, ?)'.format(USER_TABLE),
                         (user.username, user.email, hashed_password))
        await db.commit()


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create an access token.
    
    Args:
        data (dict): The data to encode in the token.
        expires_delta (Optional[timedelta], optional): The expiration time for the token.

    Returns:
        str: The encoded token.
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({'exp': expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ENCRYPTION_ALGORITHM)
    return encoded_jwt
