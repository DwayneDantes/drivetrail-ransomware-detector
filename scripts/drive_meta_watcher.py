import sqlite3
import pandas as pd
import time
import os
import json
from pathlib import Path

# --- Configuration ---
POLLING_INTERVAL = 5
EVENT_DB_PATH = os.path.join('data', 'events.db')

def find_db_path():
    """Dynamically finds the path to the DriveFS metadata database."""
    drivefs_root = Path(os.environ['LOCALAPPDATA']) / 'Google' / 'DriveFS'
    if not drivefs_root.exists(): return None
    for item in drivefs_root.iterdir():
        if item.is_dir() and item.name.isdigit():
            db_path = item / 'metadata_sqlite_db'
            if db_path.exists(): return db_path
    return None

def initialize_event_db():
    """Creates the events.db file and the 'events' table if they don't exist."""
    conn = sqlite3.connect(EVENT_DB_PATH)
    cursor = conn.cursor()
    # --- NEW: Added 'full_path' column to the table ---
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            stable_id INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            local_title TEXT,
            file_size INTEGER,
            old_local_title TEXT,
            full_path TEXT
        )
    ''')
    conn.commit()
    conn.close()
    print(f"Event database initialized at '{EVENT_DB_PATH}'")

def get_db_snapshot(conn):
    """
    Executes a more complex SQL query to get a snapshot of the current file state,
    including pre-calculating the parent relationships.
    """
    # This query now joins the items and stable_parents tables
    sql_query = """
    SELECT
        i.stable_id,
        i.local_title,
        i.modified_date,
        i.file_size,
        i.trashed,
        p.parent_stable_id
    FROM items i
    LEFT JOIN stable_parents p ON i.stable_id = p.item_stable_id
    WHERE i.is_folder = 0;
    """
    try:
        df = pd.read_sql_query(sql_query, conn)
        # We also need a map of all folder IDs to their names for path reconstruction
        folders_df = pd.read_sql_query("SELECT stable_id, local_title FROM items WHERE is_folder = 1", conn)
        folder_map = folders_df.set_index('stable_id')['local_title'].to_dict()
        return df, folder_map
    except pd.io.sql.DatabaseError as e:
        print(f"  [Warning] Database might be locked. Skipping this check. Error: {e}")
        return None, None

def reconstruct_path(stable_id, parent_id, files_df, folder_map):
    """
    Recursively reconstructs the full path for a given file.
    Note: This provides the path within Drive, not the full C:\\... path.
    """
    if pd.isna(parent_id) or parent_id not in folder_map:
        return files_df.loc[stable_id, 'local_title'] if stable_id in files_df.index else ''
    
    # This is a simplified reconstruction. A full one would trace all the way to root.
    parent_name = folder_map.get(parent_id, '...')
    file_name = files_df.loc[stable_id, 'local_title'] if stable_id in files_df.index else ''
    return os.path.join(parent_name, file_name)


def log_event(event_data):
    """Inserts an event into the SQLite events database."""
    print(f"  [EVENT LOGGED] {json.dumps(event_data, indent=2)}")
    sql_insert = "INSERT INTO events (timestamp, stable_id, event_type, local_title, file_size, old_local_title, full_path) VALUES (?, ?, ?, ?, ?, ?, ?)"
    data_tuple = (
        event_data['timestamp'], event_data['stable_id'], event_data['event_type'],
        event_data.get('local_title', ''), event_data.get('file_size', 0),
        event_data.get('old_local_title', ''), event_data.get('full_path', '')
    )
    try:
        with sqlite3.connect(EVENT_DB_PATH) as conn:
            conn.execute(sql_insert, data_tuple)
    except sqlite3.Error as e:
        print(f"  [ERROR] Could not write to SQLite database. Error: {e}")

def main():
    print("--- Starting Drive Meta Watcher (v2 - with Path Reconstruction) ---")
    initialize_event_db()
    db_path = find_db_path()
    if not db_path:
        print("ERROR: Google Drive database could not be found.")
        return

    print(f"Monitoring database: {db_path}")
    print(f"Logging events to: {EVENT_DB_PATH}")
    print(f"Polling every {POLLING_INTERVAL} seconds. Press Ctrl+C to stop.")

    prev_snapshot = None
    
    # --- We now need to process snapshots differently because of the parent_id ---
    try:
        while True:
            con_uri = f"{db_path.as_uri()}?mode=ro"
            conn = sqlite3.connect(con_uri, uri=True)
            curr_snapshot_raw, folder_map = get_db_snapshot(conn)
            conn.close()

            if curr_snapshot_raw is not None:
                # Aggregate parent IDs for each file
                curr_snapshot = curr_snapshot_raw.groupby('stable_id').agg({
                    'local_title': 'first',
                    'modified_date': 'first',
                    'file_size': 'first',
                    'trashed': 'first',
                    'parent_stable_id': lambda x: list(x)[0] if not x.isnull().all() else None
                })

                if prev_snapshot is None:
                    print("Establishing initial baseline snapshot...")
                    prev_snapshot = curr_snapshot
                else:
                    # Diffing logic remains conceptually the same...
                    # (A full implementation would be more complex, we simplify for now)
                    
                    # Find new and deleted files
                    new_ids = set(curr_snapshot.index) - set(prev_snapshot.index)
                    deleted_ids = set(prev_snapshot.index) - set(curr_snapshot.index)

                    for sid in new_ids:
                        row = curr_snapshot.loc[sid]
                        path = reconstruct_path(sid, row['parent_stable_id'], curr_snapshot, folder_map)
                        log_event({'timestamp': time.time(), 'stable_id': int(sid), 'event_type': 'CREATED',
                                   'local_title': row['local_title'], 'file_size': int(row['file_size'] or 0),
                                   'full_path': path})

                    for sid in deleted_ids:
                        row = prev_snapshot.loc[sid]
                        path = reconstruct_path(sid, row['parent_stable_id'], prev_snapshot, folder_map)
                        log_event({'timestamp': time.time(), 'stable_id': int(sid), 'event_type': 'DELETED',
                                   'local_title': row['local_title'], 'full_path': path})

                    # Find changed files
                    common_ids = set(curr_snapshot.index) & set(prev_snapshot.index)
                    for sid in common_ids:
                        prev_row = prev_snapshot.loc[sid]
                        curr_row = curr_snapshot.loc[sid]
                        path = reconstruct_path(sid, curr_row['parent_stable_id'], curr_snapshot, folder_map)
                        
                        if prev_row['local_title'] != curr_row['local_title']:
                            log_event({'timestamp': time.time(), 'stable_id': int(sid), 'event_type': 'RENAMED',
                                       'local_title': curr_row['local_title'], 'old_local_title': prev_row['local_title'],
                                       'full_path': path})
                        elif prev_row['modified_date'] != curr_row['modified_date'] or prev_row['file_size'] != curr_row['file_size']:
                            log_event({'timestamp': time.time(), 'stable_id': int(sid), 'event_type': 'MODIFIED',
                                       'local_title': curr_row['local_title'], 'file_size': int(curr_row['file_size'] or 0),
                                       'full_path': path})
                        elif prev_row['trashed'] == 0 and curr_row['trashed'] == 1:
                            log_event({'timestamp': time.time(), 'stable_id': int(sid), 'event_type': 'TRASHED',
                                       'local_title': curr_row['local_title'], 'full_path': path})

                    prev_snapshot = curr_snapshot
            time.sleep(POLLING_INTERVAL)
    except KeyboardInterrupt:
        print("\n--- Watcher stopped by user. ---")

if __name__ == "__main__":
    main()