from dotenv import load_dotenv
load_dotenv()

import os
import yaml
import sqlite3
from datetime import datetime


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_supabase_client():
    try:
        from supabase import create_client
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")
        if url and key:
            return create_client(url, key)
    except Exception:
        pass
    return None


def get_sqlite_connection():
    config = load_config()
    log_dir = config["paths"]["logs"]
    os.makedirs(log_dir, exist_ok=True)
    db_path = os.path.join(log_dir, "interactions.db")
    return sqlite3.connect(db_path)


def initialize_db():
    conn = get_sqlite_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            movie_id INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            session_id TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS recommendation_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            recommended_movie_ids TEXT NOT NULL,
            stage TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            session_id TEXT
        )
    """)

    conn.commit()
    conn.close()
    print("SQLite database initialized.")


def log_interaction(user_id, movie_id, event_type, session_id=None):
    timestamp = datetime.utcnow().isoformat()

    supabase = get_supabase_client()
    if supabase:
        try:
            supabase.table("interactions").insert({
                "user_id": int(user_id),
                "movie_id": int(movie_id),
                "event_type": event_type,
                "timestamp": timestamp,
                "session_id": session_id
            }).execute()
            return
        except Exception as e:
            print(f"Supabase log failed, falling back to SQLite: {e}")

    conn = get_sqlite_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO interactions (user_id, movie_id, event_type, timestamp, session_id)
        VALUES (?, ?, ?, ?, ?)
    """, (user_id, movie_id, event_type, timestamp, session_id))
    conn.commit()
    conn.close()


def log_recommendations(user_id, movie_ids, stage, session_id=None):
    timestamp = datetime.utcnow().isoformat()

    supabase = get_supabase_client()
    if supabase:
        try:
            supabase.table("recommendation_logs").insert({
                "user_id": int(user_id),
                "recommended_movie_ids": str(movie_ids),
                "stage": stage,
                "timestamp": timestamp,
                "session_id": session_id
            }).execute()
            return
        except Exception as e:
            print(f"Supabase log failed, falling back to SQLite: {e}")

    conn = get_sqlite_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO recommendation_logs (user_id, recommended_movie_ids, stage, timestamp, session_id)
        VALUES (?, ?, ?, ?, ?)
    """, (user_id, str(movie_ids), stage, timestamp, session_id))
    conn.commit()
    conn.close()


def get_recent_interactions(limit=20):
    supabase = get_supabase_client()
    if supabase:
        try:
            response = supabase.table("interactions").select("*").order(
                "timestamp", desc=True
            ).limit(limit).execute()
            return response.data
        except Exception as e:
            print(f"Supabase fetch failed, falling back to SQLite: {e}")

    conn = get_sqlite_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT user_id, movie_id, event_type, timestamp, session_id
        FROM interactions
        ORDER BY timestamp DESC
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [
        {"user_id": r[0], "movie_id": r[1], "event_type": r[2],
         "timestamp": r[3], "session_id": r[4]}
        for r in rows
    ]


def get_user_history(user_id):
    supabase = get_supabase_client()
    if supabase:
        try:
            response = supabase.table("interactions").select("*").eq(
                "user_id", int(user_id)
            ).order("timestamp", desc=True).execute()
            return response.data
        except Exception as e:
            print(f"Supabase fetch failed, falling back to SQLite: {e}")

    conn = get_sqlite_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT movie_id, event_type, timestamp
        FROM interactions
        WHERE user_id = ?
        ORDER BY timestamp DESC
    """, (user_id,))
    rows = cursor.fetchall()
    conn.close()
    return rows


if __name__ == "__main__":
    initialize_db()