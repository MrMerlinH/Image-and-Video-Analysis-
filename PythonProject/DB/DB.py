import sqlite3

DB_PATH = "cbvr.db"  # Replace with your actual DB path

def recreate_videos_table():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Drop old videos table if it exists
    cursor.execute("DROP TABLE shots")

    cursor.execute("""
                   CREATE TABLE shots
                   (
                       id             INTEGER PRIMARY KEY AUTOINCREMENT,
                       video_id       INTEGER,
                       start_frame    INTEGER,
                       end_frame      INTEGER,
                       keyframe_path  TEXT,
                       clip_embedding BLOB,
                       FOREIGN KEY (video_id) REFERENCES videos (id)
                   )
                   """)

    # Create the new videos table


    conn.commit()
    conn.close()
    print("✅ 'videos' table recreated successfully.")

if __name__ == "__main__":
    recreate_videos_table()
