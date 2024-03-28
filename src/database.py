import logging
import sqlite3

import aiosqlite


USER_TABLE = 'users'

DB_TABLES = [USER_TABLE]
DB_CREATION_QUERIES = {
    USER_TABLE: """
        CREATE TABLE users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            hashed_password TEXT NOT NULL
        );
    """
}


# Set up logging
logger = logging.getLogger(__name__)


# Function to check if table exists and create it if it doesn't
def create_table_if_not_exists(connection, table_name):
    assert table_name in DB_CREATION_QUERIES, f"Table '{table_name}' has no creation query!"

    cursor = connection.cursor()
    cursor.execute(f"""
    SELECT COUNT(name) FROM sqlite_master WHERE type='table' AND name='{table_name}'
    """)

    if cursor.fetchone()[0] == 1:
        logger.debug(f"Table '{table_name}' already exists.")
        
    else:
        cursor.execute(DB_CREATION_QUERIES[table_name])
        logger.info("Table 'users' created successfully.")


class DatabaseProvider:
    """A class that provides connections for each database."""

    _instance = None

    def __init__(self, db_path: str):
        """Initialize the DatabaseProvider singleton.
        
        Args:
            db_path (str): The path to the database.
        """
        self._path = db_path
        with sqlite3.connect(self._path) as connection:
            for table in DB_TABLES:
                create_table_if_not_exists(connection, table)

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            raise ValueError("The database connection has not been initialized yet!")
        return cls._instance

    @classmethod
    def initialize(cls, db_path: str):
        cls._instance = cls(db_path)
    
    def open_connection(self):
        return aiosqlite.connect(self._path)
    
    
def make_db_connection():
    """Get the database connection."""
    db_provider = DatabaseProvider.get_instance()
    return db_provider.open_connection()
