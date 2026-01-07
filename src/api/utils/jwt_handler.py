from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import jwt, JWTError

from os import getenv
from dotenv import load_dotenv
load_dotenv()

SECRET_KEY = getenv("SECRET_KEY")
ALGORITHM = getenv("ALGORITHM")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


def create_access_token(
    data: Dict[str, Any], expires_delta: Optional[timedelta] = None
) -> str:
    """
    Generates a JWT access token with an expiration time.

    Args:
        data (Dict[str, Any]): Data to encode into the token.
        expires_delta (Optional[timedelta]): Optional time delta for token expiration.

    Returns:
        str: Encoded JWT token.
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Decodes a JWT token.

    Args:
        token (str): The JWT token to decode.

    Returns:
        Optional[Dict[str, Any]]: The token payload if valid, or None if invalid or expired.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None


def get_current_user(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    """
    Retrieves the current user from the JWT token.
    Args:
        token (str): The JWT token from the request.
    Returns:
        Dict[str, Any]: The user information extracted from the token.
    Raises:
        HTTPException: If the token is invalid or expired.
    """
    payload = decode_token(token)
    
    if payload is None or payload.get("type") != "access" or "sub" not in payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
        )
    
    return {"username": payload["sub"]}


def create_refresh_token(
    data: Dict[str, Any], expires_delta: Optional[timedelta] = None
) -> str:
    """
    Generates a JWT refresh token with an expiration time.

    Args:
        data (Dict[str, Any]): Data to encode into the token.
        expires_delta (Optional[timedelta]): Optional time delta for token expiration.

    Returns:
        str: Encoded JWT refresh token.
    """
    to_encode = data.copy()
    to_encode.update({"type": "refresh"})
    expire = datetime.utcnow() + (expires_delta or timedelta(days=7))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def verify_refresh_token(token: str) -> Dict[str, Any]:
    """
    Verifies and decodes a refresh token.

    Args:
        token (str): JWT refresh token.

    Returns:
        Dict[str, Any]: Payload if valid.

    Raises:
        HTTPException: If token is invalid or not a refresh token.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token type"
            )
        if "sub" not in payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload"
            )
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )