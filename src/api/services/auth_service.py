from logging import getLogger, basicConfig, INFO
from fastapi import HTTPException, status
from passlib.context import CryptContext
from typing import Optional, Dict
from pathlib import Path
import sys
from dotenv import load_dotenv
import os

load_dotenv()

FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
ROOT_DIR = Path(__file__).resolve().parents[3]
DB_PATH = os.getenv("DATABASE_PATH")
sys.path.append(str(ROOT_DIR))

from utils.database_manager import DatabaseManager

manager = DatabaseManager(str(DB_PATH))
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

logger = getLogger(__name__)
basicConfig(level=INFO, format=FORMAT)


def verify_password(plain_password, hashed_password) -> bool:
    """
    Verifies a plain password against a hashed password.
    Args:
        plain_password (str): The plain text password to verify.
        hashed_password (str): The hashed password to compare against.
    Returns:
        bool: True if the passwords match, False otherwise.
    """
    try:
        logger.info("Verifying password.")
        password_match = pwd_context.verify(plain_password, hashed_password)
        return password_match
    except Exception as e:
        logger.error(f"Error verifying password: {e}")
        return False


def get_user(username: str) -> Optional[Dict[str, str]]:
    """
    Retrieves a user by username from the database.
    Args:
        username (str): The username of the user to retrieve.
    Returns:
        dict: A dictionary containing the user's username and hashed password, or None if not found.
    """
    try:
        logger.info(f"Retrieving user with username: {username}")
        rows = manager.select(
            "SELECT username, hashed_password FROM users WHERE username = ? LIMIT 1",
            (username,),
        )
        if not rows:
            return None
        user_info = {
            "username": rows[0]["username"],
            "hashed_password": rows[0]["hashed_password"],
        }
        return user_info
    except Exception as e:
        logger.error(f"Error retrieving user {username}: {e}")
        return None


def create_user(username: str, password: str) -> Optional[Dict[str, str | int]]:
    """
    Creates a new user with the given username and password.
    If a user with the same username already exists with the same password, returns None.
    If the username already exists with a different password, appends a suffix to the username.
    Args:
        username (str): The desired username for the new user.
        password (str): The password for the new user.
    Returns:
        dict: A dictionary containing the new user's ID and username if created successfully,
              or None if a user with the same username and password already exists.
    """
    try:
        logger.info(f"Creating user with username: {username}")
        
        existing_user = get_user(username)
        hashed = pwd_context.hash(password)
        
        if existing_user:
            if pwd_context.verify(password, existing_user["hashed_password"]):
                logger.warning(f"User with username '{username}' already exists with the same password.")
                return None
            else:
                base_username = username
                suffix = 1
                
                while get_user(f"{base_username}{suffix}") is not None:
                    suffix += 1
                
                username = f"{base_username}{suffix}"
                logger.info(f"Username '{username}' already exists, using '{username}' instead.")
        
        inserted_id = manager.insert(
            "INSERT INTO users (username, hashed_password) VALUES (?, ?)",
            (username, hashed),
        )
        return {"id": inserted_id, "username": username}
    except Exception as e:
        logger.error(f"Erro criando usuário {username}: {e}")
        return None



def authenticate_user(username: str, password: str) -> Dict[str, str]:
    """
    Authenticates a user by verifying username and password.

    Returns:
        dict: Contains 'username' if authenticated successfully.

    Raises:
        HTTPException: If authentication fails.
    """
    try:
        logger.info(f"Authenticating user with username: {username}")
        user = get_user(username)
        if not user:
            logger.warning(f"Authentication failed: user '{username}' not found.")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password",
            )

        if not verify_password(password, user["hashed_password"]):
            logger.warning(
                f"Authentication failed: incorrect password for user '{username}'."
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password",
            )

        return {"username": username}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during authentication for user '{username}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during authentication",
        )