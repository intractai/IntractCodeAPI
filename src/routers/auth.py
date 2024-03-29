from datetime import timedelta
import logging

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from jose import JWTError, jwt

from src.users import (
    authenticate_user,
    create_access_token,
    create_user,
    ENCRYPTION_ALGORITHM,
    get_user_by_username,
    oauth2_scheme,
    SECRET_KEY,
    UserCreate,
)


ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 6


logger = logging.getLogger(__name__)
router = APIRouter()


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
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ENCRYPTION_ALGORITHM])
        username: str = payload.get('sub')
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = await get_user_by_username(username)
    if user is None:
        raise credentials_exception
    return {'username': user[1], 'email': user[2]}
