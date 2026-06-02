
import os
import time
import tempfile
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st


APP_NAME = "AI School Safety Monitor"
CONTEST_NAME = "Cuộc thi Sáng tạo Thanh thiếu niên, Nhi đồng năm 2026"

STATES = {
    "Normal Activity": {
        "short": "Normal",
        "vi": "Hoạt động bình thường",
        "color": (48, 166, 78),
        "hex": "#16A34A",
        "icon": "✅",
    },
    "Abnormal Behavior Warning": {
        "short": "Abnormal",
        "vi": "Nghi ngờ đánh nhau / hành vi bất thường",
        "color": (0, 0, 255),
        "hex": "#DC2626",
        "icon": "⚠️",
    },
    "Teacher Check Required": {
        "short": "Teacher Check",
        "vi": "Cần giáo viên kiểm tra",
        "color": (0, 100, 255),
        "hex": "#F97316",
        "icon": "👨‍🏫",
    },
    "Fall Detected": {
        "short": "Fall",
        "vi": "Phát hiện té ngã",
        "color": (0, 0, 255),
        "hex": "#EF4444",
        "icon": "🚨",
    },
    "Immediate Check Required": {
        "short": "Immediate",
        "vi": "Cần kiểm tra ngay",
        "color": (0, 0, 180),
        "hex": "#991B1B",
        "icon": "🆘",
    },
}


def init_state():
    if "page" not in st.session_state:
        st.session_state.page = "Giám sát trực tiếp"
    if "counts" not in st.session_state:
        st.session_state.counts = {k: 0 for k in STATES}
        st.session_state.counts["total"] = 0
    if "events" not in st.session_state:
        st.session_state.events = []
    if "history" not in st.session_state:
        st.session_state.history = []
    if "output_path" not in st.session_state:
        st.session_state.output_path = None
    if "last_video_name" not in st.session_state:
        st.session_state.last_video_name = ""


def css():
    st.markdown(
        """
        <style>
        .stApp {
            background: #F6F8FC;
        }

        .block-container {
            max-width: 1250px;
            padding-top: 1rem;
            padding-bottom: 1rem;
            padding-left: 1.5rem;
            padding-right: 1.5rem;
        }

        #MainMenu, footer {
            visibility: hidden;
        }

        /* Fix sidebar text clipping */
        [data-testid="stSidebar"] {
            min-width: 280px !important;
            width: 280px !important;
            background: #FFFFFF;
            border-right: 1px solid #E5E7EB;
        }

        [data-testid="stSidebar"] > div:first-child {
            width: 280px !important;
            padding-left: 18px;
            padding-right: 18px;
        }

        [data-testid="stSidebar"] * {
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
        }

        .sidebar-title {
            color: #0B2E6B;
            font-size: 21px;
            font-weight: 900;
            line-height: 1.2;
            margin-bottom: 4px;
        }

        .sidebar-subtitle {
            color: #64748B;
            font-size: 13px;
            margin-bottom: 16px;
            line-height: 1.35;
        }

        .menu-item {
            background: #FFFFFF;
            color: #111827;
            border-radius: 12px;
            padding: 10px 12px;
            margin: 7px 0;
            font-size: 15px;
            font-weight: 800;
            border: 1px solid transparent;
        }

        .menu-active {
            background: #EAF2FF;
            color: #0B2E6B;
            border: 1px solid #BFDBFE;
        }

        .app-header {
            background: white;
            border: 1px solid #E5E7EB;
            border-radius: 18px;
            padding: 18px 22px;
            margin-bottom: 16px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 12px;
            flex-wrap: wrap;
        }

        .brand-title {
            color: #0B2E6B;
            font-size: 30px;
            font-weight: 900;
            line-height: 1.1;
        }

        .brand-subtitle {
            color: #64748B;
            font-size: 14px;
            margin-top: 4px;
        }

        .contest-title {
            display: inline-block;
            background: #FFF7ED;
            color: #9A3412;
            border: 1px solid #FDBA74;
            padding: 8px 12px;
            border-radius: 12px;
            font-weight: 900;
            margin-top: 10px;
            font-size: 15px;
        }

        .top-badge {
            background: #EAF2FF;
            color: #0B2E6B;
            border: 1px solid #BFDBFE;
            padding: 8px 12px;
            border-radius: 999px;
            font-weight: 800;
        }

        .main-card {
            background: white;
            border: 1px solid #E5E7EB;
            border-radius: 18px;
            padding: 18px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
        }

        .page-title {
            color: #0B2E6B;
            font-size: 24px;
            font-weight: 900;
            margin-bottom: 12px;
        }

        .state-card {
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-radius: 14px;
            padding: 12px;
            margin-bottom: 10px;
            display: grid;
            grid-template-columns: 36px 1fr 48px;
            gap: 8px;
            align-items: center;
        }

        .state-name {
            font-size: 13px;
            font-weight: 900;
            color: #111827;
            line-height: 1.2;
        }

        .state-vi {
            font-size: 12px;
            color: #6B7280;
            line-height: 1.2;
            margin-top: 2px;
        }

        .state-count {
            font-size: 23px;
            font-weight: 900;
            text-align: right;
        }

        .sidebar-help {
            background: #EEF6FF;
            color: #0B2E6B;
            border: 1px solid #BFDBFE;
            border-radius: 12px;
            padding: 12px;
            font-size: 13px;
            line-height: 1.35;
            margin-top: 16px;
        }

        .small-note {
            background: #FFFBEB;
            border: 1px solid #FDE68A;
            color: #92400E;
            border-radius: 12px;
            padding: 12px;
            margin-top: 10px;
            font-size: 14px;
            line-height: 1.4;
        }

        .stButton > button {
            background: #0B2E6B;
            color: white;
            border-radius: 12px;
            border: none;
            font-weight: 800;
            padding: 0.6rem 1rem;
        }

        .stDownloadButton > button {
            border-radius: 12px;
            font-weight: 800;
        }

        video {
            border-radius: 14px;
            max-height: 520px;
            object-fit: contain;
            background: black;
        }

        @media (max-width: 900px) {
            .brand-title {
                font-size: 24px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def sidebar():
    pages = [
        ("📹", "Giám sát "),
        ("📋", "Sự kiện"),
        ("🔔", "Cảnh báo"),
        ("🕘", "Lịch sử video"),
        ("📑", "Báo cáo"),
    ]

    st.sidebar.markdown(
        f"""
        <div class="sidebar-title">🛡️ {APP_NAME}</div>
        <div class="sidebar-subtitle">Demo cảnh báo an toàn học đường</div>
        """,
        unsafe_allow_html=True,
    )

    # Use radio for interaction, but make it less likely to clip.
    labels = [f"{icon} {name}" for icon, name in pages]
    current_index = [name for _, name in pages].index(st.session_state.page)

    selected = st.sidebar.radio(
        "Chức năng",
        labels,
        index=current_index,
        label_visibility="collapsed",
    )

    st.session_state.page = selected.split(" ", 1)[1]

    st.sidebar.markdown(
        f"""
        <div class="sidebar-help">
        <b>{CONTEST_NAME}</b><br><br>
        <b>Chức năng:</b><br>
        • Upload video demo<br>
        • Phát hiện hành vi bất thường / đánh nhau<br>
        • Phát hiện té ngã<br>
        • Hiển thị cảnh báo đỏ trên video<br>
        • Xuất báo cáo
        </div>
        """,
        unsafe_allow_html=True,
    )


def header():
    st.markdown(
        f"""
        <div class="app-header">
            <div>
                <div class="brand-title">{APP_NAME}</div>
                <div class="brand-subtitle">Phần mềm demo cảnh báo sớm hành vi bất thường và té ngã trong trường học</div>
                <div class="contest-title">{CONTEST_NAME}</div>
            </div>
            <div class="top-badge">DEMO VERSION</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def save_uploaded_file(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.close()
    return tmp.name


def draw_red_banner(frame, state):
    if state == "Normal Activity":
        return

    h, w = frame.shape[:2]
    banner_h = max(42, int(h * 0.08))
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 0, 220), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    text = "RED ALERT: " + STATES[state]["short"]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.55, min(1.0, w / 1450))
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.putText(
        frame,
        text,
        (max(10, (w - tw) // 2), int((banner_h + th) / 2) - 4),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def draw_label(frame, state, x, y):
    text = STATES[state]["short"]
    color = STATES[state]["color"]

    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.48, min(0.70, w / 1500))
    thickness = 2

    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = max(5, min(x, w - tw - 35))
    y = max(th + 12, min(y, h - 12))

    cv2.rectangle(frame, (x, y - th - 12), (x + tw + 14, y + 6), color, -1)
    cv2.putText(frame, text, (x + 7, y - 5), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_small_dashboard(frame, state, counts, idx):
    h, w = frame.shape[:2]
    x1, y1 = 10, max(50, int(h * 0.09))
    x2, y2 = min(w - 10, 520), y1 + 100

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    cv2.rectangle(frame, (x1, y1), (x2, y2), STATES[state]["color"], 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, APP_NAME, (x1 + 10, y1 + 24), font, 0.52, (7, 46, 116), 1, cv2.LINE_AA)
    cv2.putText(frame, f"State: {state}", (x1 + 10, y1 + 49), font, 0.43, (25, 25, 25), 1, cv2.LINE_AA)
    cv2.putText(
        frame,
        f"Abnormal: {counts['Abnormal Behavior Warning']} | Fall: {counts['Fall Detected']} | Immediate: {counts['Immediate Check Required']}",
        (x1 + 10, y1 + 73),
        font,
        0.40,
        (60, 60, 60),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(frame, f"Frame: {idx}", (x1 + 10, y1 + 94), font, 0.38, (70, 70, 70), 1, cv2.LINE_AA)


def motion_region(prev_gray, gray, sensitivity):
    diff = cv2.absdiff(prev_gray, gray)
    diff = cv2.GaussianBlur(diff, (7, 7), 0)
    _, thresh = cv2.threshold(diff, sensitivity, 255, cv2.THRESH_BINARY)

    kernel = np.ones((7, 7), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.dilate(thresh, kernel, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours if cv2.contourArea(c) > 500]

    if not valid:
        return None, 0.0, 0

    xs, ys, xe, ye = [], [], [], []
    total_area = 0
    for c in valid:
        x, y, w, h = cv2.boundingRect(c)
        xs.append(x)
        ys.append(y)
        xe.append(x + w)
        ye.append(y + h)
        total_area += cv2.contourArea(c)

    h, w = gray.shape[:2]
    ratio = total_area / max(1, h * w)
    return (min(xs), min(ys), max(xe), max(ye)), ratio, len(valid)


def decide_state(
    bbox,
    motion_ratio,
    num_regions,
    frame_shape,
    abnormal_threshold,
    fighting_threshold,
    fall_hold_frames,
    immediate_hold_frames,
    fall_streak,
    abnormal_streak,
):
    if bbox is None:
        return "Normal Activity", 0, max(0, abnormal_streak - 1)

    h, w = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    aspect_ratio = bw / bh
    center_y = (y1 + y2) / 2 / h

    fall_like = aspect_ratio > 1.35 and center_y > 0.52
    fighting_like = motion_ratio > fighting_threshold or (num_regions >= 2 and motion_ratio > abnormal_threshold)
    abnormal_like = motion_ratio > abnormal_threshold

    fall_streak = fall_streak + 1 if fall_like else max(0, fall_streak - 1)
    abnormal_streak = abnormal_streak + 1 if fighting_like else max(0, abnormal_streak - 1)

    if fall_streak >= immediate_hold_frames:
        return "Immediate Check Required", fall_streak, abnormal_streak

    if fall_streak >= fall_hold_frames:
        return "Fall Detected", fall_streak, abnormal_streak

    if abnormal_streak >= 12:
        return "Teacher Check Required", fall_streak, abnormal_streak

    if fighting_like or abnormal_like:
        return "Abnormal Behavior Warning", fall_streak, abnormal_streak

    return "Normal Activity", fall_streak, abnormal_streak


def process_video(input_path, output_path, config):
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise RuntimeError("Không mở được video đầu vào.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    ok, first = cap.read()
    if not ok:
        raise RuntimeError("Video không có frame hợp lệ.")

    prev_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)

    counts = {state: 0 for state in STATES}
    counts["total"] = 0

    events = []
    fall_streak = 0
    abnormal_streak = 0
    frame_index = 0

    bar = st.progress(0)
    status_box = st.empty()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_index += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        bbox, motion_ratio, num_regions = motion_region(prev_gray, gray, config["sensitivity"])

        state, fall_streak, abnormal_streak = decide_state(
            bbox=bbox,
            motion_ratio=motion_ratio,
            num_regions=num_regions,
            frame_shape=frame.shape,
            abnormal_threshold=config["abnormal_threshold"],
            fighting_threshold=config["fighting_threshold"],
            fall_hold_frames=config["fall_hold_frames"],
            immediate_hold_frames=config["immediate_hold_frames"],
            fall_streak=fall_streak,
            abnormal_streak=abnormal_streak,
        )

        counts[state] += 1
        counts["total"] = frame_index

        if state != "Normal Activity":
            events.append(
                {
                    "Thời gian": datetime.now().strftime("%H:%M:%S"),
                    "Frame": frame_index,
                    "Trạng thái": state,
                    "Mô tả": STATES[state]["vi"],
                    "Motion score": round(float(motion_ratio), 4),
                    "Số vùng chuyển động": num_regions,
                }
            )

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            color = STATES[state]["color"]
            thickness = 3 if state != "Normal Activity" else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            draw_label(frame, state, x1, y1 - 8)

        draw_red_banner(frame, state)
        draw_small_dashboard(frame, state, counts, frame_index)

        cv2.putText(
            frame,
            f"Motion: {motion_ratio:.3f} | Regions: {num_regions}",
            (18, height - 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (40, 40, 40),
            1,
            cv2.LINE_AA,
        )

        writer.write(frame)
        prev_gray = gray

        if total_frames > 0:
            bar.progress(min(frame_index / total_frames, 1.0))
        if frame_index % 25 == 0:
            status_box.info(f"Đang xử lý frame {frame_index}/{total_frames}")

    cap.release()
    writer.release()

    bar.progress(1.0)
    status_box.success("Xử lý video hoàn tất.")

    return counts, events


def state_summary_html(counts):
    html = ""
    for state, info in STATES.items():
        html += f"""
        <div class="state-card" style="border-left: 6px solid {info['hex']};">
            <div style="font-size: 24px;">{info['icon']}</div>
            <div>
                <div class="state-name">{state}</div>
                <div class="state-vi">{info['vi']}</div>
            </div>
            <div class="state-count" style="color: {info['hex']};">{counts.get(state, 0):02d}</div>
        </div>
        """
    return html


def page_monitor():
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.markdown('<div class="page-title">📹 Giám sát trực tiếp</div>', unsafe_allow_html=True)

    col_video, col_status = st.columns([2.2, 1], gap="large")

    with col_video:
        uploaded = st.file_uploader(
            "Tải video demo lên",
            type=["mp4", "avi", "mov", "mkv"],
            help="Nên dùng video ngắn 10–30 giây để xử lý nhanh.",
        )

        with st.expander("⚙️ Cấu hình phát hiện", expanded=False):
            sensitivity = st.slider("Độ nhạy chuyển động", 10, 60, 28, 1)
            abnormal_threshold = st.slider("Ngưỡng hành vi bất thường", 0.005, 0.120, 0.025, 0.005)
            fighting_threshold = st.slider("Ngưỡng đánh nhau / chuyển động mạnh", 0.010, 0.200, 0.050, 0.005)
            fall_hold_frames = st.slider("Số frame xác nhận té ngã", 1, 15, 3, 1)
            immediate_hold_frames = st.slider("Số frame cảnh báo kiểm tra ngay", 5, 60, 15, 1)

        config = {
            "sensitivity": sensitivity,
            "abnormal_threshold": abnormal_threshold,
            "fighting_threshold": fighting_threshold,
            "fall_hold_frames": fall_hold_frames,
            "immediate_hold_frames": immediate_hold_frames,
        }

        if uploaded is not None:
            st.session_state.last_video_name = uploaded.name
            input_path = save_uploaded_file(uploaded)

            st.write("**Video đầu vào**")
            st.video(input_path)

            if st.button("🚀 Bắt đầu xử lý video", type="primary"):
                output_path = os.path.join(
                    tempfile.gettempdir(),
                    f"AI_School_Safety_Result_{int(time.time())}.mp4",
                )

                try:
                    counts, events = process_video(input_path, output_path, config)
                    st.session_state.counts = counts
                    st.session_state.events = events
                    st.session_state.output_path = output_path

                    st.session_state.history.append(
                        {
                            "Thời gian xử lý": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                            "Tên video": uploaded.name,
                            "Tổng frame": counts.get("total", 0),
                            "Bất thường": counts.get("Abnormal Behavior Warning", 0),
                            "Té ngã": counts.get("Fall Detected", 0),
                            "Kiểm tra ngay": counts.get("Immediate Check Required", 0),
                        }
                    )

                    st.success("Đã xử lý xong video.")

                except Exception as e:
                    st.error(f"Lỗi: {e}")

            if st.session_state.output_path and os.path.exists(st.session_state.output_path):
                st.write("**Video sau xử lý**")
                st.video(st.session_state.output_path)

                with open(st.session_state.output_path, "rb") as f:
                    st.download_button(
                        "⬇️ Tải video kết quả",
                        data=f,
                        file_name="AI_School_Safety_Result.mp4",
                        mime="video/mp4",
                    )
        else:
            st.info("Vui lòng tải video lên để bắt đầu.")

    with col_status:
        st.markdown("### Trạng thái")
        st.markdown(state_summary_html(st.session_state.counts), unsafe_allow_html=True)

        st.markdown(
            """
            <div class="small-note">
            <b>Lưu ý:</b> Cảnh báo màu đỏ sẽ xuất hiện trực tiếp trên video khi phát hiện hành vi bất thường hoặc té ngã.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


def page_events():
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.markdown('<div class="page-title">📋 Sự kiện</div>', unsafe_allow_html=True)

    if st.session_state.events:
        df = pd.DataFrame(st.session_state.events)
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇️ Tải danh sách sự kiện CSV",
            data=csv,
            file_name="AI_School_Safety_Events.csv",
            mime="text/csv",
        )
    else:
        st.info("Chưa có sự kiện. Vui lòng xử lý video ở mục Giám sát trực tiếp.")

    st.markdown("</div>", unsafe_allow_html=True)


def page_alerts():
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.markdown('<div class="page-title">🔔 Cảnh báo</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Hành vi bất thường", st.session_state.counts.get("Abnormal Behavior Warning", 0))
    c2.metric("Té ngã", st.session_state.counts.get("Fall Detected", 0))
    c3.metric("Cần kiểm tra ngay", st.session_state.counts.get("Immediate Check Required", 0))

    st.markdown("### Chi tiết trạng thái")
    st.markdown(state_summary_html(st.session_state.counts), unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def page_history():
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.markdown('<div class="page-title">🕘 Lịch sử video</div>', unsafe_allow_html=True)

    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("Chưa có video nào được xử lý trong phiên làm việc này.")

    st.markdown("</div>", unsafe_allow_html=True)


def make_report_text():
    counts = st.session_state.counts
    total_events = len(st.session_state.events)
    video_name = st.session_state.last_video_name or "Chưa có video"

    return f"""
BÁO CÁO KẾT QUẢ DEMO
PHẦN MỀM AI SCHOOL SAFETY MONITOR

{CONTEST_NAME}

1. Thông tin video
- Tên video: {video_name}
- Thời gian tạo báo cáo: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}

2. Kết quả tổng hợp
- Tổng số frame đã xử lý: {counts.get('total', 0)}
- Normal Activity: {counts.get('Normal Activity', 0)}
- Abnormal Behavior Warning: {counts.get('Abnormal Behavior Warning', 0)}
- Teacher Check Required: {counts.get('Teacher Check Required', 0)}
- Fall Detected: {counts.get('Fall Detected', 0)}
- Immediate Check Required: {counts.get('Immediate Check Required', 0)}
- Tổng số sự kiện cảnh báo ghi nhận: {total_events}

3. Nhận xét
Phần mềm đã xử lý video đầu vào, phát hiện các vùng chuyển động bất thường,
cảnh báo nghi ngờ hành vi đánh nhau/hành vi bất thường và tình huống té ngã.
Khi phát hiện nguy cơ, hệ thống hiển thị cảnh báo màu đỏ trực tiếp trên video để giáo viên
hoặc giám thị kiểm tra kịp thời.

4. Lưu ý
Đây là phiên bản demo sử dụng xử lý ảnh và phát hiện chuyển động bằng OpenCV.
Khi triển khai thực tế, hệ thống cần được huấn luyện thêm bằng dữ liệu thực tế và kiểm thử
trong nhiều điều kiện camera, ánh sáng và góc quay khác nhau.
""".strip()


def page_report():
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.markdown('<div class="page-title">📑 Báo cáo</div>', unsafe_allow_html=True)

    report = make_report_text()
    st.text_area("Nội dung báo cáo", value=report, height=360)

    st.download_button(
        "⬇️ Tải báo cáo TXT",
        data=report.encode("utf-8-sig"),
        file_name="Bao_cao_AI_School_Safety_Monitor.txt",
        mime="text/plain",
    )

    st.markdown("</div>", unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title=APP_NAME,
        page_icon="🛡️",
        layout="wide",
    )

    init_state()
    css()
    sidebar()
    header()

    if st.session_state.page == "Giám sát trực tiếp":
        page_monitor()
    elif st.session_state.page == "Sự kiện":
        page_events()
    elif st.session_state.page == "Cảnh báo":
        page_alerts()
    elif st.session_state.page == "Lịch sử video":
        page_history()
    elif st.session_state.page == "Báo cáo":
        page_report()


if __name__ == "__main__":
    main()
