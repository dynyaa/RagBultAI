"""
Authentication module for JWT-based user authentication.

Provides password hashing, JWT token generation/validation,
and user CRUD operations for the RAG system.
"""
import logging
import secrets
import psycopg2
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

import jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from .config import Config
from .db import get_db

logger = logging.getLogger("rag.auth")

# Password hashing context using bcrypt
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# HTTP Bearer token security
security = HTTPBearer()


def hash_password(password: str) -> str:
    """Hash a plain text password using bcrypt.

    Args:
        password: Plain text password

    Returns:
        Hashed password string
    """
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain text password against a hashed password.

    Args:
        plain_password: Plain text password to verify
        hashed_password: Hashed password from database

    Returns:
        True if password matches, False otherwise
    """
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token.

    Args:
        data: Data to encode in the token (typically user_id and email)
        expires_delta: Optional custom expiration time

    Returns:
        Encoded JWT token string
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=Config.JWT_EXPIRY_HOURS)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, Config.JWT_SECRET, algorithm=Config.JWT_ALGORITHM)

    return encoded_jwt


def decode_access_token(token: str) -> Dict[str, Any]:
    """Decode and validate a JWT access token.

    Args:
        token: JWT token string

    Returns:
        Decoded token payload

    Raises:
        HTTPException: If token is invalid or expired
    """
    try:
        payload = jwt.decode(token, Config.JWT_SECRET, algorithms=[Config.JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )


def generate_api_key() -> str:
    """Generate a random API key for programmatic access.

    Returns:
        32-byte hex string (64 characters)
    """
    return secrets.token_hex(32)


def create_user(email: str, password: str) -> Dict[str, Any]:
    """Create a new user with hashed password.

    Args:
        email: User's email address
        password: Plain text password

    Returns:
        Dictionary with user_id, email, and created_at

    Raises:
        HTTPException: If email already exists
    """
    hashed_password = hash_password(password)

    with get_db() as conn:
        with conn.cursor() as cur:
            try:
                cur.execute("""
                    INSERT INTO users (email, password_hash)
                    VALUES (%s, %s)
                    RETURNING id, email, created_at
                """, (email, hashed_password))

                result = cur.fetchone()
                conn.commit()

                return {
                    "id": result[0],
                    "email": result[1],
                    "created_at": result[2]
                }
            except psycopg2.IntegrityError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already registered"
                )


def authenticate_user(email: str, password: str) -> Optional[Dict[str, Any]]:
    """Authenticate a user with email and password.

    Args:
        email: User's email address
        password: Plain text password

    Returns:
        Dictionary with user data if authentication successful, None otherwise
    """
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, email, password_hash, created_at
                FROM users
                WHERE email = %s
            """, (email,))

            result = cur.fetchone()

            if not result:
                return None

            user_id, user_email, password_hash, created_at = result

            if not verify_password(password, password_hash):
                return None

            # Update last login timestamp
            cur.execute("""
                UPDATE users
                SET last_login_at = CURRENT_TIMESTAMP
                WHERE id = %s
            """, (user_id,))
            conn.commit()

            return {
                "id": user_id,
                "email": user_email,
                "created_at": created_at
            }


def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    """Get user by ID.

    Args:
        user_id: User's ID

    Returns:
        Dictionary with user data if found, None otherwise
    """
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, email, created_at, last_login_at
                FROM users
                WHERE id = %s
            """, (user_id,))

            result = cur.fetchone()

            if not result:
                return None

            return {
                "id": result[0],
                "email": result[1],
                "created_at": result[2],
                "last_login_at": result[3]
            }


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """Get user by email.

    Args:
        email: User's email address

    Returns:
        Dictionary with user data if found, None otherwise
    """
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, email, created_at, last_login_at
                FROM users
                WHERE email = %s
            """, (email,))

            result = cur.fetchone()

            if not result:
                return None

            return {
                "id": result[0],
                "email": result[1],
                "created_at": result[2],
                "last_login_at": result[3]
            }


def get_current_user(request: Request, credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer(auto_error=False))) -> Dict[str, Any]:
    """FastAPI dependency to get the current authenticated user.

    Supports bearer token in `Authorization: Bearer ...` or an HttpOnly cookie named `auth_token`.
    """
    token = None

    # Prefer Authorization header if present
    if credentials and getattr(credentials, "credentials", None):
        token = credentials.credentials
    else:
        # Fallback to cookie
        token = request.cookies.get("auth_token")

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_access_token(token)

    user_id = payload.get("user_id")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = get_user_by_id(user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


# Optional: Dependency for routes that can work with or without authentication
def get_current_user_optional(request: Request, credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))) -> Optional[Dict[str, Any]]:
    """FastAPI dependency to optionally get the current user.

    Returns None if no token provided, otherwise validates and returns user.
    Useful for routes that enhance functionality with auth but don't require it.

    Args:
        credentials: Optional HTTP Authorization credentials

    Returns:
        Dictionary with user data or None
    """
    try:
        return get_current_user(request, credentials)
    except HTTPException:
        return None
