import sqlite3

DB_PATH = "cbvr.db"  # Replace with your actual DB path

def recreate_videos_table():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Drop old videos table if it exists
    cursor.execute("DROP TABLE videos")

    cursor.execute("""
                   CREATE TABLE videos
                   (
                       id             INTEGER PRIMARY KEY AUTOINCREMENT,
                       file_path      TEXT,
                       duration       REAL,
                       fps           REAL
                   )
                   """)

    # Create the new videos table


    conn.commit()
    conn.close()
    print("✅ 'videos' table recreated successfully.")

if __name__ == "__main__":
    recreate_videos_table()
