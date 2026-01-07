from fastapi import APIRouter, Depends, status, HTTPException, Body
from datetime import timedelta
from dotenv import load_dotenv
import os
from src.api.utils.jwt_handler import (
    create_access_token,
    get_current_user,
    verify_refresh_token,
)
from src.api.services.auth_service import authenticate_user, create_user
from src.api.schemas.auth_schema import (
    UserRequest,
    TokenResponse,
    Register,
    Login,
    Protected,
    Refresh,
)

load_dotenv()
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))

router = APIRouter(prefix="/api/v1/auth", tags=["Auth"])


@router.post("/register", **Register.docs)
def register(data: UserRequest) -> TokenResponse:
    user = create_user(data.username, data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="User already exists"
        )

    access_token = create_access_token(
        {"sub": user["username"], "type": "access"},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )

    refresh_token = create_access_token(
        {"sub": user["username"], "type": "refresh"}, expires_delta=timedelta(days=7)
    )

    return TokenResponse(access_token=access_token, refresh_token=refresh_token)


@router.post("/login", **Login.docs)
def login(data: UserRequest) -> TokenResponse:
    user = authenticate_user(data.username, data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    access_token = create_access_token(
        {"sub": user["username"], "type": "access"},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )

    refresh_token = create_access_token(
        {"sub": user["username"], "type": "refresh"}, expires_delta=timedelta(days=7)
    )

    return TokenResponse(access_token=access_token, refresh_token=refresh_token)


@router.get("/protected", **Protected.docs)
def protected_route(current_user: dict = Depends(get_current_user)) -> dict:
    return {"message": f"Authenticated access granted to {current_user['username']}!"}


@router.post("/refresh", **Refresh.docs)
def refresh_token(refresh_token: str = Body(..., embed=True)) -> TokenResponse:
    payload = verify_refresh_token(refresh_token)
    username = payload["sub"]

    new_access_token = create_access_token(
        data={"sub": username, "type": "access"},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )

    return TokenResponse(access_token=new_access_token, refresh_token=refresh_token)