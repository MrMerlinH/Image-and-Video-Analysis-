import sqlite3

DB_PATH = "cbvr.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("DROP TABLE shots_old")
# Step 1: Rename old table
cursor.execute("ALTER TABLE shots RENAME TO shots_old")

# Step 2: Create new table with AUTOINCREMENT
cursor.execute("""
    CREATE TABLE shots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        video_id INTEGER,
        start_frame INTEGER,
        end_frame INTEGER,
        keyframe_path TEXT,
        clip_embedding BLOB,
        FOREIGN KEY(video_id) REFERENCES videos(id)
    )
""")

# Step 3: Copy data (excluding id to let it auto-generate)
cursor.execute("DROP TABLE shots_old")

# Step 4: Drop old table


conn.commit()
conn.close()

print("✅ Fixed: 'id' is now AUTOINCREMENT.")
