import os
import cv2
import sqlite3

DB_PATH = "cbvr.db"
VIDEO_FOLDER = "../videos/V3C1_200"  # change if your videos are somewhere else
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov"}  # add any you want

video_files = [f for f in os.listdir(VIDEO_FOLDER)
               if os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS]
print("Detected video files:", video_files)

def get_video_metadata(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    return duration, fps

def insert_video(conn, video_id, file_path, duration, fps):
    c = conn.cursor()
    c.execute("""
        INSERT OR REPLACE INTO videos (video_id, file_path, duration, fps)
        VALUES (?, ?, ?, ?)
    """, (video_id, file_path, duration, fps))

def main():
    conn = sqlite3.connect(DB_PATH)
    video_files = [f for f in os.listdir(VIDEO_FOLDER)
                   if os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS]

    for vf in video_files:
        video_path = os.path.join(VIDEO_FOLDER, vf)
        video_id = os.path.splitext(vf)[0]  # filename without extension
        duration, fps = get_video_metadata(video_path)
        if duration is None:
            continue
        insert_video(conn, video_id, video_path, duration, fps)

    conn.commit()
    conn.close()

if __name__ == "__main__":
    main()
