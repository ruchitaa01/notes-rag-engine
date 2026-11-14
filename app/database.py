import os
from dotenv import load_dotenv
import psycopg
from pgvector.psycopg import register_vector

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/notes_rag")

def get_db_connection():
    """
    Establishes a connection to the PostgreSQL database with pgvector extension enabled.
    """
    try:
        conn = psycopg.connect(DATABASE_URL)
        # Register pgvector extension types for psycopg 3
        register_vector(conn)
        return conn
    except Exception as e:
        print(f"Database connection error: {e}")
        raise e

def init_db():
    """
    Initializes the database schema, creating the notes/chunks table and enabling pgvector.
    """
    conn = get_db_connection()
    with conn.cursor() as cur:
        # Enable pgvector extension
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
        # Create documents/chunks table for vector + hybrid search
        cur.execute("""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id SERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                embedding VECTOR(384),
                metadata JSONB
            );
        """)
        conn.commit()
    conn.close()
    print("Database initialized and pgvector extension enabled successfully!")