import streamlit as st
import cv2
import numpy as np
from time import time
import tempfile
from collections import defaultdict
from ultralytics import YOLO
from speed1 import SpeedEstimator

# Konfigurasi tampilan Streamlit
st.set_page_config(page_title="YOLOv8 Speed & Counting", layout="wide")
st.title("🚗 Perhitungan Estimasi kecepatan dan deteksi objek Kendaraan")

# Inisialisasi state Streamlit
if "paused" not in st.session_state:
    st.session_state.paused = False
if "frame_pos" not in st.session_state:
    st.session_state.frame_pos = 0
if "step" not in st.session_state:
    st.session_state.step = False
if "last_frame" not in st.session_state:
    st.session_state.last_frame = None

# Upload video
video_file = st.file_uploader("📁 Upload Video", type=["mp4", "avi"])

if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Load YOLOv8 dan estimator
    model = YOLO("yolov8n.pt")  
    line_pts = [(0, 200), (1019, 200)]
    names = model.names
    speed_obj = SpeedEstimator(reg_pts=line_pts, names=names)

    # Variabel tracking
    crossed_up, crossed_down = set(), set()
    count_up, count_down = 0, 0
    track_history = {}
    class_counts_up = defaultdict(int)
    class_counts_down = defaultdict(int)
    violations = []
    violated_ids = set()
    speeds_up = []
    speeds_down = []

    stframe = st.empty()
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Sidebar
    with st.sidebar:
        st.header("📊 Statistik Kendaraan")
        up_count_display = st.empty()
        down_count_display = st.empty()

        st.markdown("## 🎮 Kontrol Playback")
        if st.button("⏸ Pause" if not st.session_state.paused else "▶️ Lanjutkan"):
            st.session_state.paused = not st.session_state.paused

        if st.session_state.paused:
            if st.button("➡️ Step Satu Frame"):
                st.session_state.step = True

        st.session_state.frame_pos = st.slider(
            "📍 Posisi Frame", 0, total_frames - 1, st.session_state.frame_pos
        )
        cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.frame_pos)

    def get_centroid(box):
        x1, y1, x2, y2 = box
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    SPEED_LIMIT = 60

    while cap.isOpened():
        if st.session_state.paused and not st.session_state.step:
            if st.session_state.last_frame is not None:
                stframe.image(st.session_state.last_frame, channels="BGR", use_container_width=True)
            continue

        st.session_state.step = False
        cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.frame_pos)
        ret, frame = cap.read()
        if not ret:
            break

        st.session_state.frame_pos += 1
        frame = cv2.resize(frame, (640, 360))

        results = model.track(frame, persist=True, classes=[2, 3, 5, 7])
        if not results or results[0].boxes.id is None:
            st.session_state.last_frame = frame
            stframe.image(frame, channels="BGR", use_container_width=True)
            continue

        boxes = results[0].boxes.xyxy.cpu().numpy()
        clss = results[0].boxes.cls.cpu().numpy()
        ids = results[0].boxes.id.int().cpu().numpy()
        frame = speed_obj.estimate_speed(frame, results)

        for box, obj_id, cls_id in zip(boxes, ids, clss):
            cx, cy = get_centroid(box)

            if obj_id in track_history:
                prev_cx, prev_cy = track_history[obj_id]
                direction = cy - prev_cy

                if direction > 0 and prev_cy < line_pts[0][1] <= cy:
                    if obj_id not in crossed_down:
                        crossed_down.add(obj_id)
                        count_down += 1
                        class_counts_down[int(cls_id)] += 1

                elif direction < 0 and prev_cy > line_pts[0][1] >= cy:
                    if obj_id not in crossed_up:
                        crossed_up.add(obj_id)
                        count_up += 1
                        class_counts_up[int(cls_id)] += 1

            track_history[obj_id] = (cx, cy)
            speed = speed_obj.spd.get(obj_id, 0)

            if speed > 0:
                if obj_id in crossed_up:
                    speeds_up.append(speed)
                elif obj_id in crossed_down:
                    speeds_down.append(speed)

            if speed > SPEED_LIMIT and obj_id not in violated_ids:
                violated_ids.add(obj_id)
                second = st.session_state.frame_pos / fps

                cv2.putText(frame, "⚠️ Speeding!", (cx, cy + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                violations.append({
                    "id": int(obj_id),
                    "class": names[int(cls_id)],
                    "speed": round(speed, 1),
                    "direction": "down" if obj_id in crossed_down else "up",
                    "time": round(second, 2)
                })

            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(frame, f'ID:{obj_id}', (cx, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        up_count_display.metric("⬆️ Kendaraan Naik", count_up)
        down_count_display.metric("⬇️ Kendaraan Turun", count_down)
        cv2.putText(frame, f'⬆️ Up: {count_up}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f'⬇️ Down: {count_down}', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.line(frame, line_pts[0], line_pts[1], (255, 0, 0), 2)

        st.session_state.last_frame = frame
        stframe.image(frame, channels="BGR", use_container_width=True)

        # Tampilkan pelanggaran terbaru saat video berjalan
        if violations:
            last = violations[-1]
            st.markdown("### ⚠️ Pelanggaran Kecepatan Terdeteksi")
            st.warning(f"🚨 ID: {last['id']}, Class: {last['class']}, Speed: {last['speed']} km/h, "
                       f"Direction: {last['direction']}, Time: {last['time']} detik")

    cap.release()

    st.success("✅ Video processing complete.")
    st.markdown("### 🧾 Hasil Akhir")
    st.info(f"⬆️ **Total Kendaraan Naik:** {count_up}  \n⬇️ **Total Kendaraan Turun:** {count_down}")

    st.markdown("### 📈 Rata-Rata Kecepatan Kendaraan")
    if speeds_up:
        avg_up = sum(speeds_up) / len(speeds_up)
        st.success(f"⬆️ Rata-rata Kecepatan Naik: {avg_up:.2f} km/h")
    else:
        st.info("⬆️ Tidak ada kendaraan naik terdeteksi.")

    if speeds_down:
        avg_down = sum(speeds_down) / len(speeds_down)
        st.success(f"⬇️ Rata-rata Kecepatan Turun: {avg_down:.2f} km/h")
    else:
        st.info("⬇️ Tidak ada kendaraan turun terdeteksi.")

    st.markdown("### 🚨 Rekap Pelanggaran Kecepatan")
    if violations:
        for v in violations:
            st.warning(f"🚨 ID: {v['id']}, Class: {v['class']}, Speed: {v['speed']} km/h, "
                       f"Direction: {v['direction']}, Time: {v['time']} detik")
    else:
        st.info("✅ Tidak ada pelanggaran kecepatan terdeteksi.")

    st.markdown("### 📋 Detail Klasifikasi per Arah")
    def display_class_counts(class_counts, direction):
        st.subheader(f"{direction}")
        for class_id, count in class_counts.items():
            class_name = names[class_id]
            st.write(f"- {class_name}: {count}")

    display_class_counts(class_counts_up, "⬆️ Naik (Upward)")
    display_class_counts(class_counts_down, "⬇️ Turun (Downward)")
