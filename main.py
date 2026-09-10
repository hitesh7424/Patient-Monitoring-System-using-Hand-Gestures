"""
Patient Monitoring System using Hand Gestures - Multi-Hand Live Application
"""

import time
import os
from datetime import datetime
import numpy as np
import pandas as pd
import cv2
import streamlit as st
import plotly.express as px

from config import (
    CONFIDENCE_THRESHOLD,
    GESTURE_HOLD_FRAMES,
    VIDEO_FPS,
    HAND_DETECTION_CONFIDENCE,
    GESTURE_HISTORY_LIMIT,
    GESTURES,
    PATIENT_STATES,
    PATIENT_HISTORY_CSV,
)
from utils import (
    load_patient_history_df,
    load_gesture_logs,
    get_current_timestamp,
)
from gesture_detector import HandDetector, initialize_hand_detector
from gesture_classifier import classify_gesture, GestureSmoother
from patient_monitor import PatientMonitor

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Live Patient Monitoring System - Multi-Hand Vision Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Dark Mode & Medical Status Badges
st.markdown(
    """
    <style>
    .main { background-color: #0e1117; }
    .stApp { color: #ffffff; }
    .patient-status-card {
        border-radius: 12px;
        padding: 24px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.4);
        text-align: center;
        transition: all 0.3s ease;
    }
    .metric-box {
        background-color: #1e222a;
        border-radius: 10px;
        padding: 12px;
        border-left: 4px solid #007bff;
        margin-bottom: 10px;
    }
    .alert-banner {
        background-color: #dc3545;
        color: white;
        padding: 15px;
        border-radius: 8px;
        font-weight: bold;
        font-size: 1.1em;
        margin-bottom: 15px;
        text-align: center;
        box-shadow: 0 0 10px rgba(220, 53, 69, 0.7);
    }
    .stProgress > div > div > div > div {
        background-color: #007bff;
    }
    .gesture-guide-item {
        background: #1a1e24;
        padding: 8px 12px;
        border-radius: 6px;
        margin-bottom: 6px;
        font-size: 0.9em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_patient_monitor():
    """Singleton PatientMonitor state machine."""
    return PatientMonitor(confidence_threshold=CONFIDENCE_THRESHOLD)


@st.cache_resource
def get_hand_detector():
    """Singleton MediaPipe multi-hand detector (max 2 hands)."""
    return initialize_hand_detector(
        min_detection_confidence=HAND_DETECTION_CONFIDENCE,
        min_tracking_confidence=0.7
    )


def process_camera_frame(frame, detector, monitor, smoothers_dict, conf_thresh, show_landmarks, enable_smoothing):
    """
    Process camera frame with multi-hand landmark detection (Left & Right hands).
    Returns: (processed_rgb_frame, dict_of_detected_hand_gestures)
    """
    if frame is None:
        return None, {}

    # Mirror frame for intuitive viewing
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Detect up to 2 hands
    results, handedness = detector.detect_hands(frame)

    detected_hands_summary = {}

    if results and results.multi_hand_landmarks:
        # Draw skeleton overlays for all detected hands
        if show_landmarks:
            frame = detector.draw_hand_landmarks(frame, results)

        for i, raw_lm in enumerate(results.multi_hand_landmarks):
            # Determine Left / Right hand label for mirrored view
            if handedness and len(handedness) > i:
                hand_label = handedness[i]
            else:
                hand_label = "Right" if i == 0 else "Left"

            g_name, conf = classify_gesture(raw_lm)

            # Apply per-hand temporal smoothing
            if enable_smoothing and hand_label in smoothers_dict:
                g_name, conf = smoothers_dict[hand_label].add_prediction(g_name, conf)

            # Update patient state machine
            current_state, state_changed, alert = monitor.update_state(g_name, conf, hand=hand_label)

            detected_hands_summary[hand_label] = {
                "gesture": g_name,
                "confidence": conf,
                "state": current_state
            }

            # Draw HUD Box overlay for each detected hand
            box_x = 10 if i == 0 else max(10, w - 360)
            box_color = (0, 255, 127) if g_name != "Unknown" else (80, 80, 80)
            
            cv2.rectangle(frame, (box_x, 10), (box_x + 350, 90), (0, 0, 0), -1)
            cv2.rectangle(frame, (box_x, 10), (box_x + 350, 90), box_color, 2)
            cv2.putText(frame, f"{hand_label} Hand: {g_name}", (box_x + 12, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2)
            cv2.putText(frame, f"State: {current_state} ({conf*100:.0f}%)", (box_x + 12, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 200), 2)

    # Convert BGR to RGB for Streamlit rendering
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    return rgb_frame, detected_hands_summary


def main():
    st.markdown("# 🏥 Real-Time Touchless Patient Monitoring System")
    st.markdown("**Multi-Hand Computer Vision Gesture Recognition Bedside Monitor**")
    st.divider()

    # Load Singletons
    monitor = get_patient_monitor()
    detector = get_hand_detector()

    # Sidebar Controls
    st.sidebar.markdown("## ⚙️ Live Camera Controls")

    camera_index = st.sidebar.selectbox(
        "Camera Device Index",
        options=[0, 1, 2],
        format_func=lambda x: f"Camera Index {x} ({'Primary' if x==0 else 'Secondary'})",
        index=0,
    )

    conf_thresh = st.sidebar.slider(
        "Gesture Confidence Threshold",
        min_value=0.50,
        max_value=1.00,
        value=CONFIDENCE_THRESHOLD,
        step=0.05,
    )
    monitor.confidence_threshold = conf_thresh

    target_fps = st.sidebar.slider(
        "Target FPS", min_value=15, max_value=60, value=VIDEO_FPS, step=5
    )

    show_landmarks = st.sidebar.toggle("Show Hand Skeleton Landmarks", value=True)
    enable_smoothing = st.sidebar.toggle("Enable Temporal Gesture Smoothing", value=True)

    st.sidebar.divider()
    
    # PERMANENT SIDEBAR GESTURE GUIDE
    with st.sidebar.expander("📖 Gesture Guide Reference", expanded=True):
        st.markdown(
            """
            * **✅ Thumbs Up**: Patient OK (`OK`)
            * **⚠️ Thumbs Down**: In Pain (`ALERT`)
            * **🆘 Open Palm**: Help (`CALL_NURSE`)
            * **📊 Peace Sign**: Vitals (`VITALS_CHECK`)
            * **😴 Closed Fist**: Sleep (`RESTING`)
            * **👉 Point**: Severity (`POINTING`)
            """
        )

    st.sidebar.divider()
    st.sidebar.markdown("## 📁 Log & History Operations")

    # Export CSV Button
    df_history = load_patient_history_df()
    csv_data = df_history.to_csv(index=False).encode("utf-8")
    st.sidebar.download_button(
        label="📥 Download History CSV",
        data=csv_data,
        file_name=f"patient_gesture_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )

    if st.sidebar.button("🗑️ Reset Session Logs"):
        monitor.clear_history()
        st.sidebar.success("Session logs reset successfully!")
        st.rerun()

    # Main Tabs
    tab_live, tab_browser_cam, tab_history, tab_analytics, tab_guide = st.tabs(
        [
            "📹 Live Multi-Hand Stream",
            "📸 Browser Snapshot Cam",
            "📜 Gesture Log & Timeline",
            "📊 Patient Analytics",
            "📖 Gesture Guide",
        ]
    )

    # -------------------------------------------------------------
    # TAB 1: LIVE MULTI-HAND CAMERA STREAM
    # -------------------------------------------------------------
    with tab_live:
        st.markdown("### 📽️ Live Bedside Video Monitoring (Both Hands Active)")

        col_cam, col_status = st.columns([1.3, 1.0])

        with col_cam:
            run_live = st.toggle("🔴 Stream Live Camera Feed", value=True, help="Toggle live webcam feed capture.")
            video_placeholder = st.empty()
            info_placeholder = st.empty()

            # PROMINENT GESTURE REFERENCE GRID DIRECTLY UNDER CAMERA FEED
            with st.expander("📖 Gesture Quick Reference Guide (Supported Gestures)", expanded=True):
                g1, g2, g3 = st.columns(3)
                with g1:
                    st.markdown("✅ **Thumbs Up**: Patient OK")
                    st.markdown("⚠️ **Thumbs Down**: In Pain / Alert")
                with g2:
                    st.markdown("🆘 **Open Palm**: Call Nurse / Help")
                    st.markdown("📊 **Peace Sign**: Check Vitals")
                with g3:
                    st.markdown("😴 **Closed Fist**: Resting / Sleep")
                    st.markdown("👉 **Point**: Indicate Severity")

        with col_status:
            st.markdown("### 🏥 Current Patient Status")
            status_card_ph = st.empty()
            alert_banner_ph = st.empty()

            m1, m2 = st.columns(2)
            with m1:
                metric_gesture_ph = st.empty()
            with m2:
                metric_conf_ph = st.empty()

            st.markdown("#### Real-time Detection Confidence")
            progress_ph = st.empty()

            st.markdown("#### Recent Gesture Logs")
            table_ph = st.empty()

        if run_live:
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                info_placeholder.error(f"Error: Unable to open camera index {camera_index}. Please check device connection or select another index in the sidebar.")
            else:
                smoothers_dict = {
                    "Left": GestureSmoother(window_size=GESTURE_HOLD_FRAMES),
                    "Right": GestureSmoother(window_size=GESTURE_HOLD_FRAMES),
                }
                
                try:
                    while run_live:
                        ret, frame = cap.read()
                        if not ret:
                            info_placeholder.warning("Waiting for camera frames...")
                            time.sleep(0.1)
                            continue

                        rgb_frame, detected_hands = process_camera_frame(
                            frame=frame,
                            detector=detector,
                            monitor=monitor,
                            smoothers_dict=smoothers_dict,
                            conf_thresh=conf_thresh,
                            show_landmarks=show_landmarks,
                            enable_smoothing=enable_smoothing,
                        )

                        # 1. Update Video Frame
                        video_placeholder.image(rgb_frame, channels="RGB")

                        # 2. Update Side Panel Status Card
                        state_info = monitor.get_current_state()
                        state_color = state_info["badge_color"]
                        status_card_ph.markdown(
                            f"""
                            <div class="patient-status-card" style="background-color: {state_color};">
                                <h1 style="margin: 0; font-size: 3em;">{state_info['icon']} {state_info['state']}</h1>
                                <h3 style="margin-top: 10px; opacity: 0.95;">{state_info['label']}</h3>
                                <p style="margin-bottom: 0;">Updated: {state_info['timestamp'][11:19]}</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        # 3. Update Active Alert Banner
                        if monitor.active_alerts:
                            latest_alert = monitor.active_alerts[-1]
                            alert_banner_ph.markdown(
                                f"""
                                <div class="alert-banner">
                                    🚨 DISTRESS ALERT: {latest_alert['reason']}
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                        else:
                            alert_banner_ph.empty()

                        # 4. Update Metrics & Progress Bar
                        if detected_hands:
                            hand_names = list(detected_hands.keys())
                            g_summary = " | ".join([f"{h}: {detected_hands[h]['gesture']}" for h in hand_names])
                            max_conf = max([detected_hands[h]['confidence'] for h in hand_names])
                            metric_gesture_ph.metric("Live Gestures", g_summary)
                            metric_conf_ph.metric("Confidence", f"{max_conf*100:.1f}%")
                            progress_ph.progress(min(max(float(max_conf), 0.0), 1.0))
                        else:
                            metric_gesture_ph.metric("Live Gestures", f"{state_info['icon']} {state_info['last_gesture']}")
                            metric_conf_ph.metric("Confidence", f"{state_info['confidence']*100:.1f}%")
                            progress_ph.progress(min(max(float(state_info["confidence"]), 0.0), 1.0))

                        # 5. Update Recent Gesture Logs Table
                        recent_items = monitor.get_gesture_history(limit=5)
                        if recent_items:
                            table_ph.dataframe(
                                pd.DataFrame(recent_items)[["timestamp", "icon", "gesture", "patient_state", "confidence", "hand"]],
                                hide_index=True,
                            )

                        time.sleep(1.0 / target_fps)
                finally:
                    cap.release()
        else:
            info_placeholder.info("Click 'Stream Live Camera Feed' above to resume live camera monitoring.")

    # -------------------------------------------------------------
    # TAB 2: BROWSER SNAPSHOT CAM
    # -------------------------------------------------------------
    with tab_browser_cam:
        st.markdown("### 📸 Browser Camera Capture")
        st.write("Capture live frame directly through browser webcam interface:")
        
        img_file_buffer = st.camera_input("Take Live Photo for Detection")
        if img_file_buffer is not None:
            bytes_data = img_file_buffer.getvalue()
            cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

            smoothers_dict = {
                "Left": GestureSmoother(window_size=GESTURE_HOLD_FRAMES),
                "Right": GestureSmoother(window_size=GESTURE_HOLD_FRAMES),
            }
            rgb_frame, detected_hands = process_camera_frame(
                frame=cv_img,
                detector=detector,
                monitor=monitor,
                smoothers_dict=smoothers_dict,
                conf_thresh=conf_thresh,
                show_landmarks=show_landmarks,
                enable_smoothing=enable_smoothing,
            )

            st.image(rgb_frame, caption=f"Detected Hands: {detected_hands}")

    # -------------------------------------------------------------
    # TAB 3: GESTURE LOG & TIMELINE
    # -------------------------------------------------------------
    with tab_history:
        st.markdown("### 📜 Session Gesture Log")

        recent_logs = monitor.get_gesture_history(limit=50)
        if recent_logs:
            df_logs = pd.DataFrame(recent_logs)
            st.dataframe(
                df_logs[["timestamp", "icon", "gesture", "patient_state", "confidence", "hand"]],
                hide_index=True,
            )
        else:
            st.info("No gestures logged yet. Start live camera feed to detect gestures.")

        st.divider()
        st.markdown("### ⏱️ Patient State Timeline")
        df_hist = load_patient_history_df()
        if not df_hist.empty:
            fig_timeline = px.line(
                df_hist,
                x="timestamp",
                y="patient_state",
                color="gesture",
                markers=True,
                title="Patient State Timeline Progression",
                template="plotly_dark",
            )
            st.plotly_chart(fig_timeline)

    # -------------------------------------------------------------
    # TAB 4: ANALYTICS
    # -------------------------------------------------------------
    with tab_analytics:
        st.markdown("### 📊 Patient Analytics & Frequency Reports")

        df_hist = load_patient_history_df()
        if df_hist.empty:
            st.warning("No historical data recorded yet.")
        else:
            c1, c2 = st.columns(2)

            with c1:
                st.markdown("#### Gesture Frequency Bar Chart")
                counts = df_hist["gesture"].value_counts().reset_index()
                counts.columns = ["Gesture", "Count"]
                fig_bar = px.bar(
                    counts,
                    x="Gesture",
                    y="Count",
                    color="Gesture",
                    text_auto=True,
                    template="plotly_dark",
                    color_discrete_sequence=px.colors.qualitative.Bold,
                )
                st.plotly_chart(fig_bar)

            with c2:
                st.markdown("#### Patient State Share")
                state_counts = df_hist["patient_state"].value_counts().reset_index()
                state_counts.columns = ["State", "Count"]
                fig_pie = px.pie(
                    state_counts,
                    names="State",
                    values="Count",
                    hole=0.4,
                    template="plotly_dark",
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                )
                st.plotly_chart(fig_pie)

            st.divider()
            st.markdown("#### Detection Confidence Trend")
            fig_conf = px.area(
                df_hist,
                x="timestamp",
                y="confidence",
                color="gesture",
                title="Confidence Levels Over Time",
                template="plotly_dark",
            )
            st.plotly_chart(fig_conf)

    # -------------------------------------------------------------
    # TAB 5: GESTURE GUIDE
    # -------------------------------------------------------------
    with tab_guide:
        st.markdown("### 📖 Gesture Vocabulary Reference")

        g_col1, g_col2 = st.columns(2)
        cards = list(GESTURES.items())
        for idx, (g_name, g_info) in enumerate(cards):
            if g_name == "Unknown":
                continue
            target_col = g_col1 if idx % 2 == 0 else g_col2
            with target_col:
                st.markdown(
                    f"""
                    <div class="metric-box" style="border-left-color: {g_info['badge_color']};">
                        <h3>{g_info['icon']} {g_name}</h3>
                        <p><strong>Mapped State:</strong> <code>{g_info['state']}</code></p>
                        <p><strong>Meaning:</strong> {g_info['label']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


if __name__ == "__main__":
    main()
