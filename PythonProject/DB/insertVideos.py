import os
import cv2
import sqlite3

DB_PATH = "cbvr.db"
VIDEO_FOLDER = "../videos/V3C1_200"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov"}

def get_video_metadata(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    return duration, fps

def insert_video(conn, file_path, duration, fps):
    c = conn.cursor()
    c.execute("""
        INSERT INTO videos (file_path, duration, fps)
        VALUES (?, ?, ?)
    """, (file_path, duration, fps))

def main():
    conn = sqlite3.connect(DB_PATH)
    video_files = [f for f in os.listdir(VIDEO_FOLDER)
                   if os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS]

    for vf in video_files:
        video_path = os.path.join(VIDEO_FOLDER, vf)
        duration, fps = get_video_metadata(video_path)
        if duration is None:
            print(f"⚠️ Failed to read metadata for: {vf}")
            continue

        insert_video(conn, video_path, duration, fps)
        print(f"✅ Inserted: {vf} | Duration: {duration:.2f}s | FPS: {fps:.2f}")

    conn.commit()
    conn.close()
    print("📦 All video metadata inserted.")

if __name__ == "__main__":
    main()
