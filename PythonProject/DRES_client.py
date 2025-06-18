import requests

run_id = "c0c2f23f-dafe-4828-8f39-63f270688f6f"
DRES_URL = f"https://vbs.videobrowsing.org"
USERNAME = "TECHtalent09"
PASSWORD = "xhJ3T4Ct"

session = requests.Session()

# def submitAnswer(text, videoID, startTime, endTime):
#     answer = [text, videoID, "IVADL", startTime, endTime]
#     resp = session.post(f"{DRES_URL}/api/v2/submit/{evaluationId}", json={"evaluationID": evaluationId, "taskName": test["name"], "ApiClientAnswer": answer})
#     print(resp)

resp = session.post(f"{DRES_URL}/api/v2/login", json={"username": USERNAME, "password": PASSWORD})

if resp.status_code == 200:
    print("✅ Logged in to DRES")
    task_info = session.get(f"{DRES_URL}/api/v2/client/evaluation/list").json()

    evaluationId = task_info[0]["id"]
    test = session.get(f"{DRES_URL}/api/v2/client/evaluation/currentTask/{evaluationId}").json()
    print(test["name"])
    print(evaluationId)

    submission_payload = {
        "answerSets": [
            {
                "answers": [
                    {
                        "text": "",
                        "mediaItemName": "00178",
                        "mediaItemCollectionName": "IVADL",
                        "start": 0,
                        "end": 0
                    }
                ]
            }
        ]
    }

    resp = session.post(f"{DRES_URL}/api/v2/submit/{evaluationId}", json=submission_payload)
    print(resp.content)
else:
    print("❌ Login failed:", resp.text)

