import sqlite3
import json
import time
from typing import List, Dict, Any

class ScoreStore:
    def __init__(self, db_path: str = "history.db"):
        self.conn = sqlite3.connect(db_path)
        self.cur = self.conn.cursor()
        self._create_table()

    def _create_table(self):
        """Create database table with id, metadata, thought, score, and structure fields"""
        self.cur.execute("""
        CREATE TABLE IF NOT EXISTS plans (
            id TEXT PRIMARY KEY,
            metadata TEXT,
            thought TEXT,
            score REAL,
            structure TEXT  -- Stores JSON serialized string list
        )
        """)
        self.conn.commit()

    def clear_table_data(self, db_path, table_name):
        """Clear data from specified table"""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        try:
            cursor.execute(f"DELETE FROM {table_name};")
            conn.commit()
            print(f"Data in table {table_name} has been cleared")

        except sqlite3.Error as e:
            print(f"Error clearing data: {e}")
            conn.rollback()
        finally:
            conn.close()

    def add(self, metadata: Dict[str, Any], thought: List[Dict[str, Any]], score: float, structure: List[str] = None) -> str:
        """
        Store metadata(dict), thought(dict list), score and structure, and return its id (second-level timestamp)

        Args:
            metadata: Metadata dictionary
            thought: Thinking process dictionary list
            score: Score value
            structure: Structure information string list

        Returns:
            Record ID (timestamp)
        """
        ts = str(int(time.time()))  # Second-level timestamp

        # Serialize dict and dict list to JSON string
        metadata_json = json.dumps(metadata)
        thought_json = json.dumps(thought)
        # Serialize string list to JSON string
        structure_json = json.dumps(structure) if structure is not None else "[]"

        try:
            self.cur.execute(
                "INSERT INTO plans (id, metadata, thought, score, structure) VALUES (?, ?, ?, ?, ?)",
                (ts, metadata_json, thought_json, score, structure_json)
            )
            self.conn.commit()
        except sqlite3.IntegrityError:
            # If a record already exists for this second, append an auto-increment suffix to id
            ts = ts + "_" + str(int(time.time() * 1000) % 1000)  # Append millisecond suffix
            self.cur.execute(
                "INSERT INTO plans (id, metadata, thought, score, structure) VALUES (?, ?, ?, ?, ?)",
                (ts, metadata_json, thought_json, score, structure_json)
            )
            self.conn.commit()
        return ts

    def top_k(self, k: int = 10):
        """
        Return top k records sorted by score, metadata, thought, and structure are deserialized

        Args:
            k: Number of records to return

        Returns:
            Tuple list containing (metadata, thought, score, structure)
        """
        self.cur.execute(
            "SELECT metadata, thought, score, structure FROM plans ORDER BY score DESC LIMIT ?", (k,)
        )
        rows = self.cur.fetchall()
        # Deserialize JSON strings to original data types
        rpns = []
        scores = []
        thoughts = []
        for row in rows:
            metadata, thought, score, structure = row
            thoughts.append(json.loads(thought))
            rpns.append(json.loads(structure))
            scores.append(score)
        return thoughts,rpns,scores

    def load(self, id_: str):
        """
        Load record by id, metadata, thought, and structure are deserialized

        Args:
            id_: Record ID

        Returns:
            Tuple containing (metadata, thought, score, structure), or None if not exists
        """
        self.cur.execute("SELECT metadata, thought, score, structure FROM plans WHERE id=?", (id_,))
        row = self.cur.fetchone()
        if row:
            metadata, thought, score, structure = row
            return (
                json.loads(metadata),
                json.loads(thought),
                score,
                json.loads(structure)  # Deserialize to string list
            )
        return None

    def all(self):
        """
        Return all records, sorted by score descending, metadata, thought, and structure are deserialized

        Returns:
            Tuple list containing (metadata, thought, score, structure)
        """
        self.cur.execute("SELECT metadata, thought, score, structure FROM plans ORDER BY score DESC")
        rows = self.cur.fetchall()
        # Deserialize JSON strings to original data types
        result = []
        for row in rows:
            metadata, thought, score, structure = row
            result.append((
                json.loads(metadata),
                json.loads(thought),
                score,
                json.loads(structure)  # Deserialize to string list
            ))
        return result

    def close(self):
        """Close database connection"""
        self.conn.close()