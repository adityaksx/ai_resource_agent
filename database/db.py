import sqlite3
import json
from datetime import datetime


DB_PATH = "database/resources.db"


def get_connection():
    return sqlite3.connect(DB_PATH)


def init_db():

    conn = get_connection()
    cursor = conn.cursor()

    # Main resources table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS resources (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source TEXT,
        url TEXT,
        title TEXT,
        raw_input TEXT,
        raw_data TEXT,
        cleaned_data TEXT,
        llm_output TEXT,
        files TEXT,
        status TEXT,
        error TEXT,
        created_at TEXT
    )
    """)

    conn.commit()
    conn.close()


# -------------------------
# Save new resource
# -------------------------

def save_resource(
    source,
    url,
    title=None,
    raw_input=None,
    raw_data=None,
    cleaned_data=None,
    llm_output=None,
    files=None,
    status="processed",
    error=None
):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO resources
        (source,url,title,raw_input,raw_data,cleaned_data,llm_output,files,status,error,created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            source,
            url,
            title,
            json.dumps(raw_input) if raw_input else None,
            json.dumps(raw_data) if raw_data else None,
            json.dumps(cleaned_data) if cleaned_data else None,
            llm_output,
            json.dumps(files) if files else None,
            status,
            error,
            datetime.utcnow().isoformat()
        )
    )

    conn.commit()
    conn.close()


# -------------------------
# Fetch past resources
# -------------------------

def get_resources(limit=20):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT id,source,url,title,created_at FROM resources ORDER BY id DESC LIMIT ?",
        (limit,)
    )

    rows = cursor.fetchall()

    conn.close()

    return rows


# -------------------------
# Fetch full resource
# -------------------------

def get_resource(resource_id):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM resources WHERE id=?",
        (resource_id,)
    )

    row = cursor.fetchone()

    conn.close()

    return row