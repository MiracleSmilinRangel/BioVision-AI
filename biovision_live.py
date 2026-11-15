
import streamlit as st
import streamlit.components.v1 as components
from streamlit_autorefresh import st_autorefresh
import cv2
import av
import mediapipe as mp
import numpy as np
import pandas as pd
import time
import math

from collections import deque
from scipy.signal import butter, sosfiltfilt, detrend, find_peaks
from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
    VideoProcessorBase
)

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="BioVision AI",
    page_icon="🧬",
    layout="wide"
)

FACE_MODEL = "models/face_landmarker.task"
POSE_MODEL = "models/pose_landmarker_lite.task"

MODULES = {
    "Blink & Fatigue Dynamics": {
        "icon": "👁️",
        "description": "Monitor eye activity and blink patterns."
    },
    "Respiration Rate": {
        "icon": "🫁",
        "description": "Estimate breathing rate from upper-body movement."
    },
    "Heart Rate (rPPG)": {
        "icon": "🫀",
        "description": "Estimate heart rate using facial colour variations."
    }
}



st.markdown("""
<style>
:root {
    --bg: #05090d;
    --panel: rgba(13, 25, 30, 0.94);
    --line: rgba(53, 242, 173, 0.38);
    --neon: #35f2ad;
    --text: #e8fff6;
    --muted: #a6c7bb;
}

/* =====================================
   GLOBAL NEON GREEN DARK THEME
===================================== */

.stApp,
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse at 78% 8%, rgba(0,210,137,.11), transparent 32%),
        radial-gradient(ellipse at 8% 90%, rgba(0,160,120,.08), transparent 34%),
        #05090d !important;
    color: var(--text) !important;
}

[data-testid="stHeader"] {
    background: rgba(4,9,13,.92) !important;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}

/* =====================================
   SIDEBAR
===================================== */

[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#081419,#050d11) !important;
    border-right: 1px solid rgba(53,242,173,.35);
    box-shadow: 8px 0 35px rgba(0,255,170,.06);
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span {
    color: #d9eee6 !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: rgba(53,242,173,.10) !important;
    border-radius: 10px;
}

/* =====================================
   BRAND AND HEADINGS
===================================== */

.brand {
    color: #35f2ad !important;
    font-size: 34px;
    font-weight: 850;
    letter-spacing: -1px;
    text-shadow: 0 0 10px rgba(53,242,173,.65),
                 0 0 25px rgba(0,201,130,.35);
}

.subtitle {
    color: #9bc9b8 !important;
    letter-spacing: 3px;
    font-size: 11px;
    font-weight: 650;
    text-transform: uppercase;
}

h1, h2, h3, h4 {
    color: #eafff5 !important;
}

p, li, label {
    color: #d9eee6 !important;
}

small,
[data-testid="stCaptionContainer"] {
    color: #a6c7bb !important;
}

/* =====================================
   GLASS PANELS
===================================== */

.panel {
    background: linear-gradient(
        145deg,
        rgba(17,42,35,.96),
        rgba(10,20,27,.98)
    ) !important;

    border: 1px solid rgba(53,242,173,.38) !important;
    padding: 22px;
    border-radius: 18px;

    box-shadow:
        inset 0 1px 0 rgba(255,255,255,.04),
        0 0 18px rgba(0,255,170,.08),
        0 12px 35px rgba(0,0,0,.25);
}

/* =====================================
   METRIC CARDS
===================================== */

[data-testid="stMetric"] {
    background: linear-gradient(
        145deg,
        rgba(17,42,37,.96),
        rgba(10,20,27,.98)
    ) !important;

    border: 1px solid rgba(53,242,173,.38) !important;
    border-radius: 15px !important;
    padding: 17px 19px !important;

    box-shadow:
        0 0 16px rgba(0,255,170,.08),
        inset 0 1px 0 rgba(255,255,255,.04);
}

[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p {
    color: #b8d9cb !important;
}

[data-testid="stMetricValue"],
[data-testid="stMetricValue"] div {
    color: #35f2ad !important;
    text-shadow: 0 0 12px rgba(53,242,173,.28);
}

/* =====================================
   ALL ALERTS - INFO, SUCCESS, WARNING,
   ERROR - APPLY THROUGHOUT THE APP
===================================== */

/* Main alert container */
[data-testid="stAlert"],
.stAlert {
    background: linear-gradient(
        135deg,
        #102820 0%,
        #0a1c19 100%
    ) !important;

    background-color: #102820 !important;

    border: 1px solid #35f2ad !important;
    border-radius: 12px !important;

    box-shadow:
        0 0 10px rgba(53,242,173,.18),
        0 0 22px rgba(0,201,130,.10) !important;

    opacity: 1 !important;
}

/* Remove pale backgrounds inside alerts */
[data-testid="stAlert"] > div,
[data-testid="stAlert"] > div > div,
[data-testid="stAlert"] [data-testid="stMarkdownContainer"] {
    background: transparent !important;
    background-color: transparent !important;
}

/* Force readable text in every alert */
[data-testid="stAlert"] p,
[data-testid="stAlert"] span,
[data-testid="stAlert"] li,
[data-testid="stAlert"] div,
[data-testid="stAlert"] [data-testid="stMarkdownContainer"] {
    color: #e8fff6 !important;
    opacity: 1 !important;
    text-shadow: none !important;
}

/* Alert icons */
[data-testid="stAlert"] svg {
    color: #35f2ad !important;
    fill: #35f2ad !important;
}

/* Sidebar alerts */
[data-testid="stSidebar"] [data-testid="stAlert"] {
    background: #102820 !important;
    border: 1px solid #35f2ad !important;
}

[data-testid="stSidebar"] [data-testid="stAlert"] p,
[data-testid="stSidebar"] [data-testid="stAlert"] span {
    color: #35f2ad !important;
}

/* =====================================
   CUSTOM STATUS MESSAGE
===================================== */

.status-message {
    background: linear-gradient(135deg,#102820,#0a1c19) !important;
    color: #35f2ad !important;
    border: 1px solid #35f2ad !important;
    border-radius: 12px;
    padding: 14px 18px;
    box-shadow: 0 0 14px rgba(53,242,173,.15);
    font-weight: 600;
}

/* =====================================
   BUTTONS
===================================== */

div.stButton > button {
    background: linear-gradient(100deg,#00a978,#35f2ad) !important;
    color: #03120d !important;
    border: 1px solid #8affd0 !important;
    border-radius: 10px !important;
    font-weight: 800 !important;
    min-height: 44px;

    box-shadow: 0 0 13px rgba(0,230,150,.25);
    transition: all 160ms ease-in-out;
}

div.stButton > button:hover {
    background: linear-gradient(100deg,#22d99b,#8affd0) !important;
    color: #03120d !important;
    box-shadow: 0 0 22px rgba(53,242,173,.48);
    transform: translateY(-1px);
}

/* =====================================
   RADIO BUTTONS AND INPUTS
===================================== */

[data-testid="stRadio"] label,
[data-testid="stSelectbox"] label {
    color: #d9eee6 !important;
}

input,
textarea,
[data-baseweb="select"] > div {
    background-color: #0b191d !important;
    color: #e8fff6 !important;
    border-color: rgba(53,242,173,.35) !important;
}

/* =====================================
   PROGRESS BAR
===================================== */

[data-testid="stProgressBar"] > div {
    background: #18332b !important;
    border-radius: 20px !important;
}

[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg,#00a978,#35f2ad) !important;
    box-shadow: 0 0 12px rgba(53,242,173,.45);
}

/* =====================================
   CHARTS
===================================== */

[data-testid="stLineChart"],
[data-testid="stVegaLiteChart"],
[data-testid="stArrowVegaLiteChart"] {
    background: #0b191d !important;
    border: 1px solid rgba(53,242,173,.32) !important;
    border-radius: 15px !important;
    padding: 10px;
    box-shadow: 0 0 17px rgba(0,255,170,.06);
}

/* =====================================
   DIVIDERS AND FOOTER
===================================== */

hr {
    border-color: rgba(53,242,173,.25) !important;
}

footer {
    visibility: hidden;
}

/* =====================================
   LIGHT THEME OVERRIDE
   If Streamlit is switched to light mode
===================================== */

html[data-theme="light"] .stApp,
html[data-theme="light"] [data-testid="stAppViewContainer"] {
    background: #f2f7f5 !important;
    color: #172b24 !important;
}

html[data-theme="light"] h1,
html[data-theme="light"] h2,
html[data-theme="light"] h3,
html[data-theme="light"] h4,
html[data-theme="light"] p,
html[data-theme="light"] li,
html[data-theme="light"] label {
    color: #263f34 !important;
    text-shadow: none !important;
}

html[data-theme="light"] .brand {
    color: #008b60 !important;
    text-shadow: none !important;
}

html[data-theme="light"] .panel,
html[data-theme="light"] [data-testid="stMetric"] {
    background: #ffffff !important;
    border: 1px solid rgba(0,130,85,.28) !important;
}

html[data-theme="light"] [data-testid="stMetricValue"] {
    color: #008b60 !important;
    text-shadow: none !important;
}

html[data-theme="light"] [data-testid="stSidebar"] {
    background: #e8f2ed !important;
}

html[data-theme="light"] [data-testid="stSidebar"] p,
html[data-theme="light"] [data-testid="stSidebar"] label {
    color: #263f34 !important;
}

html[data-theme="light"] [data-testid="stAlert"] {
    background: #e5f5ec !important;
    border: 1px solid #008b60 !important;
}

html[data-theme="light"] [data-testid="stAlert"] p,
html[data-theme="light"] [data-testid="stAlert"] span,
html[data-theme="light"] [data-testid="stAlert"] div {
    color: #174b32 !important;
}

</style>
""", unsafe_allow_html=True)






# ---------------- SIGNAL FUNCTIONS ----------------

def calculate_rate(data, mode):

    if mode == "heart":
        minimum = 250
        low, high = 0.75, 3.0
    else:
        minimum = 300
        low, high = 0.1, 0.5

    if len(data) < minimum:
        return None

    times = np.array([x[0] for x in data])
    values = np.array([x[1] for x in data])

    uniform_times = np.arange(
        times[0], times[-1], 0.1
    )

    if len(uniform_times) < minimum:
        return None

    signal = np.interp(uniform_times, times, values)
    signal = detrend(signal)

    try:
        sos = butter(
            2,
            [low, high],
            btype="bandpass",
            fs=10,
            output="sos"
        )

        filtered = sosfiltfilt(sos, signal)

        if mode == "heart":

            window = np.hanning(len(filtered))
            spectrum = np.abs(
                np.fft.rfft(filtered * window)
            )
            frequencies = np.fft.rfftfreq(
                len(filtered), d=0.1
            )

            valid = (
                (frequencies >= low) &
                (frequencies <= high)
            )

            if not np.any(valid):
                return None

            indices = np.where(valid)[0]
            peak = indices[
                np.argmax(spectrum[valid])
            ]

            rate = frequencies[peak] * 60

        else:

            peaks, _ = find_peaks(
                filtered,
                distance=12,
                prominence=max(
                    np.std(filtered) * 0.25, 0.001
                )
            )

            if len(peaks) < 3:
                return None

            intervals = np.diff(peaks) * 0.1
            rate = 60 / np.median(intervals)

        if 6 <= rate <= 180:
            return round(float(rate), 1)

    except Exception:
        return None

    return None


def distance(p1, p2):
    return math.hypot(
        p1.x - p2.x,
        p1.y - p2.y
    )


def eye_ratio(landmarks, indices):

    points = [landmarks[i] for i in indices]

    vertical1 = distance(points[1], points[5])
    vertical2 = distance(points[2], points[4])
    horizontal = distance(points[0], points[3])

    if horizontal == 0:
        return 0

    return (vertical1 + vertical2) / (2 * horizontal)


# ---------------- WEBCAM PROCESSOR ----------------

class BioVisionProcessor(VideoProcessorBase):

    def __init__(self, selected_module):

        self.module = selected_module

        self.face_detector = None
        self.pose_detector = None
        self.error = None

        self.start_time = time.monotonic()
        self.last_sample = 0
        self.last_update = 0

        self.samples = deque(maxlen=450)
        # Frozen video detection
        self.previous_frame_small = None
        self.identical_frame_count = 0
        self.video_frozen = False

        self.blink_count = 0
        self.blink_times = deque(maxlen=120)
        self.eye_closed = False
        self.closed_start = 0
        self.ear = 0
        self.ear_history = deque(maxlen=300)
        # Rolling eye-closure history for an experimental PERCLOS indicator.
        self.eye_state_history = deque()
        self.perclos = 0.0
        self.fatigue_status = "Waiting for face detection"

        self.heart_rate = None
        self.breathing_rate = None

        # Keep a short history of valid estimates to reduce display jitter.
        # A median is less sensitive to occasional noisy readings.
        self.heart_rate_history = deque(maxlen=5)
        self.breathing_rate_history = deque(maxlen=5)

        try:

            if self.module in [
                "Blink & Fatigue Dynamics",
                "Heart Rate (rPPG)"
            ]:

                options = mp.tasks.vision.FaceLandmarkerOptions(
                    base_options=mp.tasks.BaseOptions(
                        model_asset_path=FACE_MODEL
                    ),
                    running_mode=mp.tasks.vision.RunningMode.VIDEO,
                    num_faces=1
                )

                self.face_detector = (
                    mp.tasks.vision.FaceLandmarker
                    .create_from_options(options)
                )

            if self.module == "Respiration Rate":

                options = mp.tasks.vision.PoseLandmarkerOptions(
                    base_options=mp.tasks.BaseOptions(
                        model_asset_path=POSE_MODEL
                    ),
                    running_mode=mp.tasks.vision.RunningMode.VIDEO,
                    num_poses=1
                )

                self.pose_detector = (
                    mp.tasks.vision.PoseLandmarker
                    .create_from_options(options)
                )

        except Exception as error:
            self.error = str(error)

    def recv(self, frame):

        image = frame.to_ndarray(format="bgr24")

        if self.error:
            cv2.putText(
                image,
                "Model error - check terminal",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

            return av.VideoFrame.from_ndarray(
                image, format="bgr24"
            )

        # -------- FROZEN FRAME DETECTION --------

        now = time.monotonic()

        small_frame = cv2.resize(
            image,
            (32, 18)
        )

        gray_frame = cv2.cvtColor(
            small_frame,
            cv2.COLOR_BGR2GRAY
        )


        if self.previous_frame_small is not None:

            difference = np.mean(
                cv2.absdiff(
                    gray_frame,
                    self.previous_frame_small
                )
            )

            if difference < 0.15:
                self.identical_frame_count += 1
            else:
                self.identical_frame_count = 0
                self.video_frozen = False

            # Approximately 2 seconds at 10 FPS
            if self.identical_frame_count >= 20:
                self.video_frozen = True

        self.previous_frame_small = gray_frame.copy()

        if self.video_frozen:

            cv2.putText(
                image,
                "VIDEO FROZEN - SIGNAL COLLECTION PAUSED",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

            return av.VideoFrame.from_ndarray(
                image,
                format="bgr24"
            )

        timestamp = int(
            (now - self.start_time) * 1000
        )

        height, width = image.shape[:2]

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        # -------- BLINK MODULE --------

        if self.module == "Blink & Fatigue Dynamics":

            result = self.face_detector.detect_for_video(
                mp_image, timestamp
            )

            status = "Face not detected"

            if result.face_landmarks:

                landmarks = result.face_landmarks[0]

                left_eye = [33, 160, 158, 133, 153, 144]
                right_eye = [362, 385, 387, 263, 373, 380]

                left = eye_ratio(landmarks, left_eye)
                right = eye_ratio(landmarks, right_eye)

                self.ear = (left + right) / 2
                self.ear_history.append(
                    (now - self.start_time, self.ear)
                )

                if self.ear < 0.20:

                    status = "Eyes Closed"

                    if not self.eye_closed:
                        self.eye_closed = True
                        self.closed_start = now

                else:

                    status = "Eyes Open"

                    if self.eye_closed:

                        duration = now - self.closed_start

                        if 0.08 <= duration <= 0.6:
                            self.blink_count += 1
                            self.blink_times.append(now)

                    self.eye_closed = False

                # Keep eye-open/closed samples from the latest 30 seconds.
                self.eye_state_history.append((now, self.eye_closed))
                cutoff = now - 30.0
                while self.eye_state_history and self.eye_state_history[0][0] < cutoff:
                    self.eye_state_history.popleft()

                if self.eye_state_history:
                    closed_samples = sum(
                        1 for _, closed in self.eye_state_history if closed
                    )
                    self.perclos = 100.0 * closed_samples / len(self.eye_state_history)

                closure_duration = (
                    now - self.closed_start if self.eye_closed else 0.0
                )
                if self.eye_closed and closure_duration >= 2.0:
                    self.fatigue_status = "Prolonged eye closure"
                elif self.perclos >= 20.0:
                    self.fatigue_status = "Possible fatigue"
                else:
                    self.fatigue_status = "Normal"
            else:
                self.fatigue_status = "Face not detected"

            cv2.putText(
                image,
                f"Status: {status}",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 180),
                2
            )
            cv2.putText(
                image,
                f"Fatigue: {self.fatigue_status}",
                (20, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 180),
                2
            )

            cv2.putText(
                image,
                f"Blink Count: {self.blink_count}",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 180),
                2
            )

            cv2.putText(
                image,
                f"EAR: {self.ear:.3f}",
                (20, 135),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        # -------- HEART RATE MODULE --------

        elif self.module == "Heart Rate (rPPG)":

            result = self.face_detector.detect_for_video(
                mp_image, timestamp
            )

            if result.face_landmarks:

                landmarks = result.face_landmarks[0]

                xs = [p.x for p in landmarks]
                ys = [p.y for p in landmarks]

                min_x = max(0, int(min(xs) * width))
                max_x = min(width, int(max(xs) * width))
                min_y = max(0, int(min(ys) * height))
                max_y = min(height, int(max(ys) * height))

                fw = max_x - min_x
                fh = max_y - min_y

                if fw > 0 and fh > 0:

                    x1 = int(min_x + fw * 0.30)
                    x2 = int(min_x + fw * 0.70)
                    y1 = int(min_y + fh * 0.08)
                    y2 = int(min_y + fh * 0.25)

                    x1 = max(0, x1)
                    x2 = min(width, x2)
                    y1 = max(0, y1)
                    y2 = min(height, y2)

                    if x2 > x1 and y2 > y1:

                        roi = image[y1:y2, x1:x2]

                        green = float(
                            np.mean(roi[:, :, 1])
                        )

                        if now - self.last_sample >= 0.1:

                            self.samples.append(
                                (now - self.start_time, green)
                            )

                            self.last_sample = now

                        cv2.rectangle(
                            image,
                            (x1, y1),
                            (x2, y2),
                            (0, 255, 100),
                            2
                        )

            if now - self.last_update >= 1:

                candidate_rate = calculate_rate(
                    self.samples, "heart"
                )

                # Update only when a valid estimate is available.
                # Otherwise retain the most recent valid displayed value.
                if candidate_rate is not None:
                    self.heart_rate_history.append(candidate_rate)
                    self.heart_rate = round(
                        float(np.median(self.heart_rate_history)), 1
                    )

                self.last_update = now

            display = (
                f"Estimated Heart Rate: {self.heart_rate} BPM"
                if self.heart_rate is not None
                else "Collecting heart signal..."
            )

            cv2.putText(
                image,
                display,
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 180),
                2
            )

            cv2.putText(
                image,
                f"Samples: {len(self.samples)}",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        # -------- RESPIRATION MODULE --------

        elif self.module == "Respiration Rate":

            result = self.pose_detector.detect_for_video(
                mp_image, timestamp
            )

            if result.pose_landmarks:

                pose = result.pose_landmarks[0]

                ls, rs = pose[11], pose[12]
                lh, rh = pose[23], pose[24]

                visibility = [
                    ls.visibility, rs.visibility,
                    lh.visibility, rh.visibility
                ]

                if min(visibility) > 0.5:

                    sx = (ls.x + rs.x) / 2
                    sy = (ls.y + rs.y) / 2
                    hy = (lh.y + rh.y) / 2

                    sw = abs(ls.x - rs.x)

                    x1 = max(0, int((sx - sw * 0.65) * width))
                    x2 = min(width, int((sx + sw * 0.65) * width))
                    y1 = max(0, int(sy * height))
                    y2 = min(
                        height,
                        int((sy + (hy - sy) * 0.55) * height)
                    )

                    if x2 > x1 and y2 > y1:

                        roi = image[y1:y2, x1:x2]

                        gray = cv2.cvtColor(
                            roi, cv2.COLOR_BGR2GRAY
                        )

                        signal = float(np.mean(gray))

                        if now - self.last_sample >= 0.1:

                            self.samples.append(
                                (now - self.start_time, signal)
                            )

                            self.last_sample = now

                        cv2.rectangle(
                            image,
                            (x1, y1),
                            (x2, y2),
                            (255, 180, 0),
                            2
                        )

            if now - self.last_update >= 1:

                candidate_rate = calculate_rate(
                    self.samples, "breath"
                )

                # Median smoothing reduces sudden jumps between estimates.
                if candidate_rate is not None:
                    self.breathing_rate_history.append(candidate_rate)
                    self.breathing_rate = round(
                        float(np.median(self.breathing_rate_history)), 1
                    )

                self.last_update = now

            display = (
                f"Respiration: {self.breathing_rate} breaths/min"
                if self.breathing_rate is not None
                else "Collecting breathing signal..."
            )

            cv2.putText(
                image,
                display,
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 180),
                2
            )

            cv2.putText(
                image,
                f"Samples: {len(self.samples)}",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        cv2.putText(
            image,
            "BIOVISION AI",
            (20, height - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 180),
            2
        )

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24"
        )

    def close(self):

        if self.face_detector:
            self.face_detector.close()

        if self.pose_detector:
            self.pose_detector.close()


# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.markdown(
        '<div class="brand">🧬 BioVision AI</div>',
        unsafe_allow_html=True
    )

    st.caption("INTELLIGENT MONITORING SYSTEM")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        ["Overview", "Monitoring", "About"]
    )

    st.divider()

    st.markdown("### SYSTEM")
    st.markdown(
        '<div class="status-message">✓ Application Ready</div>',
        unsafe_allow_html=True
    )


# ---------------- HEADER ----------------

st.markdown(
    '<div class="brand">BIOVISION AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-POWERED CONTACTLESS HEALTH MONITORING'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ---------------- OVERVIEW ----------------

if page == "Overview":

    st.title("Welcome to BioVision AI")

    st.write(
        "Select an intelligent monitoring feature "
        "to begin webcam-based analysis."
    )

    cols = st.columns(3)

    for col, (name, info) in zip(cols, MODULES.items()):

        with col:

            st.markdown(f"""
            <div class="panel">
                <h1>{info['icon']}</h1>
                <h3 style="color:#45efb0">{name}</h3>
                <p>{info['description']}</p>
            </div>
            """, unsafe_allow_html=True)

    st.write("")

    st.markdown(
        '<div class="status-message">'
        ' Navigate to Monitoring to select and start a module.'
        '</div>',
        unsafe_allow_html=True
    )


# ---------------- MONITORING ----------------

elif page == "Monitoring":

    st.title("Monitoring Center")

    st.write("Choose one module to begin.")

    selected = st.radio(
        "SELECT MODULE",
        list(MODULES.keys()),
        format_func=lambda x: (
            f"{MODULES[x]['icon']}  {x}"
        )
    )

    st.markdown(f"""
    <div class="panel">
        <h3 style="color:#45efb0">
            {MODULES[selected]['icon']} {selected}
        </h3>
        <p>{MODULES[selected]['description']}</p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    if st.button("Proceed to Live Monitoring"):

        st.session_state["proceed"] = True
        st.session_state["selected_module"] = selected

    if st.session_state.get("proceed", False):

        active = st.session_state.get(
            "selected_module", selected
        )

        st.divider()

        st.subheader(f"Live Monitoring: {active}")

        st.info(
            "Press START in the camera panel to begin. "
            "Allow time for signal collection."
        )

        
        ctx = webrtc_streamer(
            key=f"biovision-{active}",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=lambda: BioVisionProcessor(active),
            media_stream_constraints={
                "video": True,
                "audio": False
            },
            video_html_attrs={
                "autoPlay": True,
                "controls": False,
                "muted": True,
                "style": {"width": "100%"}
            },
            async_processing=True
        )

        # Refresh dashboard every second
        st_autorefresh(
            interval=1000,
            key=f"refresh-{active}"
        )

        st.markdown("### LIVE MEASUREMENTS")
        st.caption(
            "Heart-rate and respiration displays use a rolling median of recent "
            "valid estimates to reduce sudden fluctuations. Values remain estimates."
        )

        processor = ctx.video_processor
        if "session_summary" not in st.session_state:
            st.session_state.session_summary = None

        if processor is not None:
            elapsed = time.monotonic() - processor.start_time

            if active == "Blink & Fatigue Dynamics":
                st.session_state.session_summary = {
                    "module": active,
                    "duration": round(elapsed, 1),
                    "total_blinks": processor.blink_count,
                    "ear": round(processor.ear, 3),
                    "perclos": round(processor.perclos, 1),
                    "fatigue_status": processor.fatigue_status
                }

            elif active == "Respiration Rate":
                st.session_state.session_summary = {
                    "module": active,
                    "duration": round(elapsed, 1),
                    "rate": processor.breathing_rate,
                    "samples": len(processor.samples)
                }

            elif active == "Heart Rate (rPPG)":
                st.session_state.session_summary = {
                    "module": active,
                    "duration": round(elapsed, 1),
                    "rate": processor.heart_rate,
                    "samples": len(processor.samples)
                }

        if processor is not None:

            if processor.error:
                st.error(f"Model error: {processor.error}")

            elif active == "Blink & Fatigue Dynamics":

                current_time = time.monotonic()

                recent_blinks = [
                    t for t in processor.blink_times
                    if current_time - t <= 60
                ]

                c1, c2, c3 = st.columns(3)

                with c1:
                    st.metric("Total Blinks", processor.blink_count)

                with c2:
                    st.metric("Blink Rate", f"{len(recent_blinks)} / min")

                with c3:
                    st.metric("Eye Aspect Ratio", round(processor.ear, 3))

                st.markdown("#### Fatigue Status")
                st.metric("PERCLOS (30-second window)", f"{processor.perclos:.1f}%")

                st.markdown("**PERCLOS percentage guide**")
                fatigue_ranges = pd.DataFrame({
                    "Status": ["Normal", "Possible fatigue", "Prolonged eye closure"],
                    "PERCLOS (%)": ["0–19.9%", "20–100%", "Not percentage-based"],
                    "Condition": [
                        "PERCLOS is below 20%",
                        "PERCLOS is 20% or higher",
                        "Eyes remain closed continuously for 2 seconds or more"
                    ]
                })
                st.table(fatigue_ranges)

                if processor.fatigue_status == "Normal":
                    st.success("Normal")
                elif processor.fatigue_status == "Possible fatigue":
                    st.warning("Possible fatigue")
                elif processor.fatigue_status == "Prolonged eye closure":
                    st.error("Prolonged eye closure")
                else:
                    st.info("Face not detected — adjust your position.")
                st.caption(
                    "Experimental indicator only: possible fatigue is flagged at "
                    "PERCLOS ≥ 20%, and prolonged eye closure at ≥ 2 seconds. "
                    "These thresholds are illustrative, not medically validated."
                )
                st.markdown("#### Eye Tracking Status")

                if processor.ear > 0:
                    st.success("Face tracking active")
                else:
                    st.warning("Face not detected. Adjust your position.")
                st.markdown("#### Live Eye Activity")

                if processor.ear_history:
                    ear_data = pd.DataFrame(
                        list(processor.ear_history),
                        columns=["Time", "Eye Aspect Ratio"]
                    ).set_index("Time")

                    st.line_chart(ear_data, height=250)
                else:
                    st.info("Waiting for eye activity data...")

                if processor.ear < 0:
                
                    st.warning("Waiting for face detection")

            elif active == "Respiration Rate":

                c1, c2 = st.columns(2)

                with c1:
                    st.metric(
                        "Respiration Rate",
                        f"{processor.breathing_rate} breaths/min"
                        if processor.breathing_rate is not None
                        else "Collecting..."
                    )

                with c2:
                    st.metric("Signal Samples", len(processor.samples))
                st.markdown("#### Signal Collection Progress")

                progress = min(len(processor.samples) / 300, 1.0)

                st.progress(progress)

                if progress < 1:
                    st.info(f"Collecting signal: {len(processor.samples)}/300 samples")
                else:
                    st.success("Minimum sample window reached")
                st.markdown("#### Live Respiration Signal")

                if processor.samples:
                    signal_data = pd.DataFrame(
                        list(processor.samples),
                        columns=["Time", "Signal"]
                    ).set_index("Time")

                    st.line_chart(signal_data, height=250)
                else:
                    st.info("Waiting for respiration signal...")

                if processor.breathing_rate is not None:
                    st.success("Breathing signal estimated")
                else:
                    st.info("Keep your upper body visible and remain still.")

            elif active == "Heart Rate (rPPG)":

                c1, c2 = st.columns(2)

                with c1:
                    st.metric(
                        "Estimated Heart Rate",
                        f"{processor.heart_rate} BPM"
                        if processor.heart_rate is not None
                        else "Collecting..."
                    )

                with c2:
                    st.metric("Signal Samples", len(processor.samples))
                st.markdown("#### Signal Collection Progress")

                progress = min(len(processor.samples) / 250, 1.0)

                st.progress(progress)

                if progress < 1:
                    st.info(f"Collecting signal: {len(processor.samples)}/250 samples")
                else:
                    st.success("Minimum sample window reached")
                st.markdown("#### Live Heart Rate Signal")

                if processor.samples:
                    signal_data = pd.DataFrame(
                        list(processor.samples),
                        columns=["Time", "Signal"]
                    ).set_index("Time")

                    st.line_chart(signal_data, height=250)
                else:
                    st.info("Waiting for heart signal...")

                if processor.heart_rate is not None:
                    st.success("Heart signal estimated")
                else:
                    st.info("Keep your face visible, well-lit and relatively still.")

        else:
            st.info("Start the webcam to view live measurements.")

        st.divider()
        st.markdown("### SESSION SUMMARY")

        summary = st.session_state.session_summary

        if summary and summary["module"] == active:

            st.caption("Latest recorded session snapshot")

            st.metric(
                "Session Duration",
                f"{summary['duration']} seconds"
            )

            if active == "Blink & Fatigue Dynamics":

                c1, c2 = st.columns(2)

                c1.metric("Total Blinks", summary["total_blinks"])
                c2.metric("Final Eye Aspect Ratio", summary["ear"])
                st.metric("Final Fatigue Status", summary["fatigue_status"])
                st.metric("Final PERCLOS", f"{summary['perclos']}%")

            elif active == "Respiration Rate":

                c1, c2 = st.columns(2)

                c1.metric(
                    "Last Estimated Respiration",
                    f"{summary['rate']} breaths/min"
                    if summary["rate"] is not None else "Unavailable"
                )

                c2.metric("Samples Collected", summary["samples"])

            elif active == "Heart Rate (rPPG)":

                c1, c2 = st.columns(2)

                c1.metric(
                    "Last Estimated Heart Rate",
                    f"{summary['rate']} BPM"
                    if summary["rate"] is not None else "Unavailable"
                )

                c2.metric("Samples Collected", summary["samples"])

            st.caption(
                "Experimental estimates only. "
                "This summary is not a medical report."
            )


        if st.button("Back to Module Selection"):

            st.session_state["proceed"] = False
            st.rerun()


# ---------------- ABOUT ----------------

elif page == "About":

    st.title("About BioVision AI")

    st.write("""
    BioVision AI is an experimental computer vision
    application designed to explore contactless monitoring
    using webcam images and signal processing.
    """)

    st.markdown("""
    **Core Technologies**
    - Python
    - OpenCV
    - MediaPipe
    - NumPy
    - SciPy
    - Streamlit
    - Streamlit-WebRTC
    """)

    st.markdown(
            '<div class="status-message">'
            ' Measurements are experimental estimates, not medical readings or diagnoses.'
            '</div>',
            unsafe_allow_html=True
    )        


# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "BIOVISION AI | COMPUTER VISION & SIGNAL PROCESSING"
)
