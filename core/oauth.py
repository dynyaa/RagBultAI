import logging
import os
import httpx
from urllib.parse import quote
from typing import Optional
import google.oauth2.id_token
import google.auth.transport.requests

from .config import Config
from .auth import create_access_token
from .db import get_db

logger = logging.getLogger("rag.oauth")


def build_google_auth_url(state: Optional[str] = None) -> str:
    base = "https://accounts.google.com/o/oauth2/v2/auth"
    params = {
        "client_id": Config.GOOGLE_CLIENT_ID,
        "response_type": "code",
        "scope": "openid email profile",
        "redirect_uri": Config.OAUTH_REDIRECT_URI,
        "access_type": "offline",
        "prompt": "consent"
    }
    if state:
        params["state"] = state

    # build query string
    qs = "&".join([f"{k}={quote(str(v))}" for k, v in params.items()])
    return f"{base}?{qs}"


def exchange_code_for_tokens(code: str) -> dict:
    token_url = "https://oauth2.googleapis.com/token"
    data = {
        "code": code,
        "client_id": Config.GOOGLE_CLIENT_ID,
        "client_secret": Config.GOOGLE_CLIENT_SECRET,
        "redirect_uri": Config.OAUTH_REDIRECT_URI,
        "grant_type": "authorization_code",
    }
    resp = httpx.post(token_url, data=data, timeout=10.0)
    resp.raise_for_status()
    return resp.json()


def verify_id_token(id_token: str) -> dict:
    request = google.auth.transport.requests.Request()
    try:
        # Standard verification (raises ValueError on failures)
        info = google.oauth2.id_token.verify_oauth2_token(
            id_token, request, Config.GOOGLE_CLIENT_ID
        )
        return info
    except ValueError as e:
        msg = str(e)
        # Common cause: small clock skew between client and Google's servers
        if "Token used too early" in msg or "used before" in msg or "iat" in msg:
            try:
                # Allow small clock skew (10 seconds) and retry verification
                info = google.oauth2.id_token.verify_oauth2_token(
                    id_token,
                    request,
                    Config.GOOGLE_CLIENT_ID,
                    clock_skew_in_seconds=10,
                )
                return info
            except Exception:
                # Re-raise original error if retry fails
                raise
        # Not a clock-skew issue — re-raise
        raise


def create_or_get_user_from_google(id_info: dict) -> dict:
    """
    Insert or fetch a user by Google subject ID (sub).

    Note: existing schema may require a non-null password_hash; to remain compatible
    we insert an empty string for password_hash if necessary. The migration will
    later make password_hash nullable.
    """
    sub = id_info.get("sub")
    email = id_info.get("email")
    name = id_info.get("name")

    with get_db() as conn:
        with conn.cursor() as cur:
            # 1) If a user already has this google_id, return it
            cur.execute("SELECT id, email, created_at FROM users WHERE google_id = %s", (sub,))
            row = cur.fetchone()
            if row:
                return {"id": row[0], "email": row[1], "created_at": row[2]}

            # 2) If a user exists with this email, link the google_id and return
            cur.execute("SELECT id, email, created_at FROM users WHERE email = %s", (email,))
            row_email = cur.fetchone()
            if row_email:
                user_id = row_email[0]
                try:
                    cur.execute("UPDATE users SET google_id = %s WHERE id = %s", (sub, user_id))
                    conn.commit()
                except Exception:
                    # If update fails due to race, try to select again
                    conn.rollback()
                return {"id": row_email[0], "email": row_email[1], "created_at": row_email[2]}

            # 3) Otherwise create a fresh user linked to this google_id
            cur.execute(
                "INSERT INTO users (email, google_id, password_hash, created_at) VALUES (%s, %s, %s, CURRENT_TIMESTAMP) RETURNING id, email, created_at",
                (email, sub, "")
            )
            new = cur.fetchone()
            conn.commit()
            return {"id": new[0], "email": new[1], "created_at": new[2]}
