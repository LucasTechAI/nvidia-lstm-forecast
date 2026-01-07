from pydantic import BaseModel


class UserRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class Register:
    docs = {
        "response_model": TokenResponse,
        "responses": {
            200: {
                "description": "User registered successfully. Returns a JWT access and refresh token.",
                "content": {
                    "application/json": {
                        "example": {
                            "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                            "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                            "token_type": "bearer",
                        }
                    }
                },
            },
            400: {
                "description": "User registration failed due to existing username.",
                "content": {
                    "application/json": {"example": {"detail": "User already exists"}}
                },
            },
        },
    }


class Login:
    docs = {
        "response_model": TokenResponse,
        "responses": {
            200: {
                "description": "Successful authentication. Returns a JWT access and refresh token.",
                "content": {
                    "application/json": {
                        "example": {
                            "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                            "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                            "token_type": "bearer",
                        }
                    }
                },
            },
            401: {
                "description": "Invalid credentials.",
                "content": {
                    "application/json": {
                        "example": {"detail": "Invalid username or password"}
                    }
                },
            },
        },
    }


class Protected:
    docs = {
        "responses": {
            200: {
                "description": "Authenticated access granted.",
                "content": {
                    "application/json": {"example": {"message": "Access granted"}}
                },
            },
            401: {
                "description": "Unauthorized access.",
                "content": {
                    "application/json": {
                        "example": {"detail": "Invalid authentication credentials"}
                    }
                },
            },
        }
    }


class Refresh:
    docs = {
        "response_model": TokenResponse,
        "responses": {
            200: {
                "description": "Token refreshed successfully.",
                "content": {
                    "application/json": {
                        "example": {
                            "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                            "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                            "token_type": "bearer",
                        }
                    }
                },
            },
            401: {
                "description": "Unauthorized access.",
                "content": {
                    "application/json": {
                        "example": {"detail": "Invalid authentication credentials"}
                    }
                },
            },
        },
    }