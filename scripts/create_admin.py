"""
Create Admin User Script

Creates a new user account from the command line.
Useful for setting up the first user in a fresh deployment.

Usage:
    python scripts/create_admin.py --email admin@example.com --password secretpassword
    python scripts/create_admin.py --email user@company.com --password mypass123
"""
import sys
import os
import argparse

# Add parent directory to path to import from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg2
from core.config import Config

from core.auth import create_user


def main():
    parser = argparse.ArgumentParser(
        description="Create a new user account",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/create_admin.py --email admin@example.com --password secretpass
  python scripts/create_admin.py --email user@company.com --password mypass123

Security Notes:
  - Use a strong password (min 8 characters recommended)
  - Store credentials securely (password manager, env vars, etc.)
  - Consider using environment variables instead of command line args
        """
    )

    parser.add_argument(
        "--email",
        required=True,
        help="Email address for the new user"
    )

    parser.add_argument(
        "--password",
        required=False,
        help="Password for the new user (min 8 characters recommended). Omit with --no-password."
    )

    parser.add_argument(
        "--no-password",
        action="store_true",
        help="Create user without a password (useful for OAuth/passwordless admin)."
    )

    parser.add_argument(
        "--google-id",
        required=False,
        help="Optional Google subject ID to link to this user (creates oauth-linked user)."
    )

    args = parser.parse_args()

    # Validate inputs
    if "@" not in args.email:
        print("❌ Error: Invalid email address")
        sys.exit(1)

    if not args.no_password:
        if not args.password:
            print("❌ Error: --password is required unless --no-password is set")
            sys.exit(1)

        if len(args.password) < 8:
            print("⚠️  Warning: Password is less than 8 characters (not recommended)")
            response = input("Continue anyway? (y/N): ")
            if response.lower() != 'y':
                print("Aborted.")
                sys.exit(0)

    # Create user
    print(f"\nCreating user: {args.email}")

    try:
        if args.no_password:
            # Insert directly into DB with empty password_hash (migration will allow NULL)
            conn = psycopg2.connect(Config.PG_CONN)
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO users (email, password_hash, google_id, created_at) VALUES (%s, %s, %s, CURRENT_TIMESTAMP) RETURNING id, email, created_at",
                    (args.email, "", args.google_id if args.google_id else None)
                )
                new = cur.fetchone()
                conn.commit()
                print("✅ User created successfully (passwordless)!")
                print(f"\nUser Details:")
                print(f"  ID: {new[0]}")
                print(f"  Email: {new[1]}")
                print(f"  Created: {new[2]}")
        else:
            user = create_user(args.email, args.password)
            # Optionally link google_id
            if args.google_id:
                conn = psycopg2.connect(Config.PG_CONN)
                with conn.cursor() as cur:
                    cur.execute("UPDATE users SET google_id = %s WHERE id = %s", (args.google_id, user["id"]))
                    conn.commit()
            print("✅ User created successfully!")
            print(f"\nUser Details:")
            print(f"  ID: {user['id']}")
            print(f"  Email: {user['email']}")
            print(f"  Created: {user['created_at']}")

    except Exception as e:
        print(f"❌ Failed to create user: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
