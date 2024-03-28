from datetime import datetime, timedelta
import logging
from typing import Optional, Union

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr

from src.database import make_db_connection, USER_TABLE


SECRET_KEY = '0GHa94H6pg89hlgvoJYeG+1LTKkjysoxYKIumfXirog='
ALGORITHM = 'HS256'
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 6


logger = logging.getLogger(__name__)
router = APIRouter()


oauth2_scheme = OAuth2PasswordBearer(tokenUrl='token')
pwd_context = CryptContext(schemes=['bcrypt'], deprecated='auto')


### Pydantic models ###

class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str


class UserLogin(BaseModel):
    username: str
    password: str


### Routes ###

@router.post('/register/')
async def register(user: UserCreate):
    await create_user(user)
    return {'message': "User created successfully."}


@router.post('/token')
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={'WWW-Authenticate': 'Bearer'},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    username = user[1]
    access_token = create_access_token(
        data={'sub': username}, expires_delta=access_token_expires
    )
    return {'access_token': access_token, 'token_type': 'bearer'}

@router.get('/users/me/')
async def read_users_me(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={'WWW-Authenticate': 'Bearer'},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get('sub')
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = await get_user_by_username(username)
    if user is None:
        raise credentials_exception
    return {'username': user[1], 'email': user[2]}


### Helper functions ###

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create an access token.
    
    Args:
        data (dict): The data to encode in the token.
        expires_delta (Optional[timedelta], optional): The expiration time for the token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({'exp': expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


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


async def create_user(user: UserCreate):
    hashed_password = pwd_context.hash(user.password)
    async with make_db_connection() as db:
        await db.execute(f"INSERT INTO {USER_TABLE} (username, email, hashed_password) VALUES (?, ?, ?)",
                         (user.username, user.email, hashed_password))
        await db.commit()
