from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import mediapipe as mp
from datetime import datetime

app = FastAPI()

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- SESSION MEMORY ----------------
session_state = {
    "gaze_off_frames": 0,
    "total_frames": 0,
    "continuous_off": 0,
    "max_continuous_off": 0,
}

# ---------------- ROOT ----------------
@app.get("/")
def home():
    return {"message": "AI Server Running 🚀"}

# ---------------- ANALYZE ----------------
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    contents = await file.read()

    # Convert to image
    npimg = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if frame is None:
        return {"error": "Invalid image"}

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 🔥 Load MediaPipe INSIDE request (fixes crash)
    mp_face = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
    mp_pose = mp.solutions.pose.Pose()

    face_result = mp_face.process(frame_rgb)
    pose_result = mp_pose.process(frame_rgb)

    gaze_off = False
    posture_bad = False

    # ---------------- FACE / GAZE ----------------
    if face_result.multi_face_landmarks:
        lm = face_result.multi_face_landmarks[0].landmark

        left_eye = lm[33]
        right_eye = lm[263]
        nose = lm[1]

        eye_center_y = (left_eye.y + right_eye.y) / 2
        eye_center_x = (left_eye.x + right_eye.x) / 2

        # Looking down
        if eye_center_y > 0.55:
            gaze_off = True

        # Looking sideways
        if abs(left_eye.x - right_eye.x) > 0.25:
            gaze_off = True

        # Head turn
        if abs(nose.x - eye_center_x) > 0.1:
            gaze_off = True

    else:
        gaze_off = True  # no face detected

    # ---------------- POSTURE ----------------
    if pose_result.pose_landmarks:
        lm = pose_result.pose_landmarks.landmark
        shoulder_diff = abs(lm[11].y - lm[12].y)

        if shoulder_diff > 0.08:
            posture_bad = True

    # ---------------- CLOSE MEDIAPIPE ----------------
    mp_face.close()
    mp_pose.close()

    # ---------------- SESSION TRACK ----------------
    session_state["total_frames"] += 1

    if gaze_off:
        session_state["gaze_off_frames"] += 1
        session_state["continuous_off"] += 1
    else:
        session_state["continuous_off"] = 0

    session_state["max_continuous_off"] = max(
        session_state["max_continuous_off"],
        session_state["continuous_off"]
    )

    attention_score = 100 - (
        session_state["gaze_off_frames"] / max(1, session_state["total_frames"])
    ) * 100

    return {
        "gaze_off": gaze_off,
        "posture_bad": posture_bad,
        "attention_score": round(attention_score, 2),
        "continuous_off_frames": session_state["continuous_off"],
        "max_continuous_off": session_state["max_continuous_off"]
    }

# ---------------- FINAL REPORT ----------------
@app.get("/final-report")
def final_report():
    total = session_state["total_frames"]
    off = session_state["gaze_off_frames"]

    attention = 100 - (off / max(1, total)) * 100

    cheating = (
        attention < 60 or
        session_state["max_continuous_off"] > 15
    )

    scores = {
        "attention": round(attention, 2),
        "integrity": 40 if cheating else 90,
        "posture": 70,
        "confidence": 75
    }

    final_score = round(
        (scores["attention"] * 0.4 +
         scores["confidence"] * 0.3 +
         scores["posture"] * 0.2 +
         scores["integrity"] * 0.1) / 10,
        1
    )

    tips = []

    if attention < 70:
        tips.append("Maintain consistent eye contact with camera")

    if session_state["max_continuous_off"] > 10:
        tips.append("Avoid looking away for long durations")

    if cheating:
        tips.append("Avoid using external devices during discussion")

    if not tips:
        tips.append("Good performance. Maintain consistency.")

    return {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "scores": scores,
        "final_score": final_score,
        "cheating": cheating,
        "tips": tips
    }