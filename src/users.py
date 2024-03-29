from datetime import datetime, timedelta
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


oauth2_scheme = OAuth2PasswordBearer(tokenUrl='token')
pwd_context = CryptContext(schemes=['bcrypt'], deprecated='auto')


### Session tracker ###

class SessionTracker:
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(
            cls: 'SessionTracker',
            max_active_sessions: int = 100,
            eviction_interval: int = 300,
            max_session_idle_time: int = 600
        ) -> 'SessionTracker':
        """Get the singleton instance of the session tracker.

        Args:
            max_active_sessions (int, optional): The maximum number of users allowed. Defaults to 100.
            eviction_interval (int, optional): The interval to check for inactive users. Defaults to 300.
            max_session_idle_time (int, optional): The threshold to evict inactive users. Defaults to 600.

        Returns:
            SessionTracker: The singleton instance of the session tracker.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(max_active_sessions, eviction_interval, max_session_idle_time)
        return cls._instance

    def __init__(self, max_active_sessions: int, eviction_interval: int, max_session_idle_time: int):
        """Initialize the session tracker."""
        if SessionTracker._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            SessionTracker._instance = self
            self.max_active_sessions = max_active_sessions
            self.eviction_interval = eviction_interval
            self.max_session_idle_time = max_session_idle_time
            self.user_activity = {}
            self._start_eviction_thread()

    def _start_eviction_thread(self):
        """Start the eviction thread to remove inactive users from the session tracker.
        
        The eviction thread runs every `eviction_interval` seconds and removes users who have not been active for
        `max_session_idle_time` seconds.
        """
        def eviction_process():
            while True:
                current_time = time.time()
                with SessionTracker._lock:
                    for user, last_active in list(self.user_activity.items()):
                        if current_time - last_active > self.max_session_idle_time:
                            del self.user_activity[user]
                time.sleep(self.eviction_interval)

        thread = threading.Thread(target=eviction_process)
        thread.daemon = True
        thread.start()

    def update_user_session(self, username: str) -> bool:
        """Update a user's session by updating their last activity time.
        
        Args:
            username (str): The username to update.
            
        Returns:
            bool: True if the user was updated, False if the user was not updated.
        """
        with SessionTracker._lock:
            if username in self.user_activity:
                self.user_activity[username] = time.time()
                return True
            elif len(self.user_activity) < self.max_active_sessions:
                self.user_activity[username] = time.time()
                return True
            else:
                return False


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


def validate_user_session(username: str = Depends(get_current_user)):
    """Validate's a user's token and updates their session.
    
    This function should be a dependency for any route that requires a user to be logged in.
    """
    sess_tracker = SessionTracker.get_instance()
    sess_tracker.update_user_session(username)
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
    """Creates a new user by inserting their information into the database."""
    hashed_password = pwd_context.hash(user.password)
    async with make_db_connection() as db:
        await db.execute(f"INSERT INTO {USER_TABLE} (username, email, hashed_password) VALUES (?, ?, ?)",
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
