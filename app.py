
import os, time, tempfile
import cv2
import numpy as np
import streamlit as st

# =========================================================
# AI SCHOOL SAFETY MONITOR - V4
# Focus:
# - Detect fighting / abnormal behavior demo
# - Detect fall demo
# - Red warning on video
#
# Note:
# This is a lightweight demo using OpenCV motion heuristics.
# For real deployment, train a YOLO/Action Recognition model.
# =========================================================

STATES = {
    "Normal Activity": ("Normal", "Hoạt động bình thường", "#22A65A", (48,166,78), "✅"),
    "Abnormal Behavior Warning": ("Fighting / Abnormal", "Nghi ngờ đánh nhau / hành vi bất thường", "#E23232", (0,0,255), "⚠️"),
    "Teacher Check Required": ("Teacher Check", "Cần giáo viên kiểm tra", "#D97706", (0,80,255), "👨‍🏫"),
    "Fall Detected": ("Fall Detected", "Phát hiện té ngã", "#E23232", (0,0,255), "🚨"),
    "Immediate Check Required": ("Immediate Check", "Cần kiểm tra ngay", "#B91C1C", (0,0,180), "🆘"),
}

def css():
    st.markdown("""
    <style>
    .stApp{background:linear-gradient(180deg,#fff,#f4f8fe)}
    .block-container{max-width:1650px;padding-top:1rem}
    #MainMenu, footer{visibility:hidden}
    .title{text-align:center;color:#073276;font-size:38px;font-weight:900}
    .decor{text-align:center;color:#0A4EA3;letter-spacing:6px;margin-bottom:14px}
    .shell{border:2px solid #073276;border-radius:20px;background:white;box-shadow:0 15px 40px #07327615;overflow:hidden}
    .topbar{display:flex;justify-content:space-between;align-items:center;padding:14px 18px;border-bottom:1px solid #D5E4F7;gap:12px;flex-wrap:wrap}
    .brand{display:flex;align-items:center;gap:12px;color:#073276;font-size:24px;font-weight:900}
    .shield{width:44px;height:44px;border:3px solid #0A4EA3;border-radius:12px 12px 18px 18px;display:flex;align-items:center;justify-content:center;background:#F3F8FF;color:#0A4EA3;font-weight:900}
    .pill{font-size:13px;background:#E9F3FF;border:1px solid #B7D7FF;border-radius:9px;padding:6px 10px;color:#0A4EA3}
    .grid{display:grid;grid-template-columns:180px 1fr 300px;gap:14px;padding:16px}
    .nav{border-right:1px solid #E6EEF9;padding-right:10px}
    .navitem{padding:10px;border-radius:11px;margin-bottom:9px;font-size:14px;font-weight:650}
    .active{background:#E9F3FF;border:1px solid #C8DEFF;color:#0A4EA3;font-weight:900}
    .card{border:2px solid #073276;border-radius:15px;overflow:hidden;background:#F9FCFF}
    .cardhead{background:#073276;color:white;padding:11px 15px;font-weight:900;display:flex;justify-content:space-between;flex-wrap:wrap}
    .cardbody{padding:10px;background:linear-gradient(135deg,#F8FBFF,#EAF3FF)}
    .upload{border:2px dashed #A7C6EF;background:#F7FBFF;border-radius:14px;padding:16px;text-align:center;margin-bottom:10px}
    .control{display:flex;gap:14px;padding:10px 14px;color:#073276;font-weight:800;border-top:1px solid #DCE8F7;flex-wrap:wrap}
    .live{background:#E6F1FF;border-radius:8px;padding:5px 10px;color:#0A4EA3}
    .panel{border:1px solid #C9DDF7;border-radius:15px;background:white;overflow:hidden}
    .ptitle{background:#0A4EA3;color:white;text-align:center;font-size:18px;font-weight:900;padding:13px}
    .pbody{padding:12px 10px}
    .state{border-radius:13px;padding:10px;margin-bottom:10px;display:grid;grid-template-columns:34px 1fr 45px;gap:8px;align-items:center;background:#F9FBFF;border:1px solid #E4EAF5}
    .sname{font-size:12px;font-weight:900;line-height:1.2;color:#102033}
    .svi{font-size:11px;color:#5B6575;line-height:1.25}
    .scount{font-size:22px;font-weight:900;text-align:right}
    .note{margin:16px auto 0;border:1px solid #C6DBF6;background:#F5FAFF;border-radius:16px;padding:13px 20px;color:#16315C;font-size:15px}
    video{border-radius:12px;max-height:520px;object-fit:contain;background:#000}
    .rednote{background:#FFF4F4;border:1px solid #F29A9A;color:#991B1B;border-radius:12px;padding:12px;margin:10px 0}
    @media(max-width:1100px){.grid{grid-template-columns:1fr}.nav{border-right:none;border-bottom:1px solid #E6EEF9}}
    </style>
    """, unsafe_allow_html=True)

def save_uploaded_file(f):
    tmp=tempfile.NamedTemporaryFile(delete=False,suffix=os.path.splitext(f.name)[1])
    tmp.write(f.read()); tmp.close(); return tmp.name

def draw_label(img, state, x, y):
    short, _, _, color, _ = STATES[state]
    h,w=img.shape[:2]
    font=cv2.FONT_HERSHEY_SIMPLEX
    scale=max(.50,min(.70,w/1500)); thick=2
    (tw,th),_=cv2.getTextSize(short,font,scale,thick)
    x=max(5,min(x,w-tw-30)); y=max(th+12,min(y,h-10))
    # Red label for every warning, green only for normal.
    cv2.rectangle(img,(x,y-th-12),(x+tw+14,y+6),color,-1)
    cv2.putText(img,short,(x+7,y-5),font,scale,(255,255,255),thick,cv2.LINE_AA)

def draw_big_red_warning(img, state):
    """
    Draw a visible red warning banner at top when state is warning.
    """
    if state == "Normal Activity":
        return
    h,w=img.shape[:2]
    banner_h = max(42, int(h*0.075))
    overlay=img.copy()
    cv2.rectangle(overlay,(0,0),(w,banner_h),(0,0,220),-1)
    cv2.addWeighted(overlay,.82,img,.18,0,img)
    text = "RED ALERT: " + STATES[state][0]
    font=cv2.FONT_HERSHEY_SIMPLEX
    scale=max(.55,min(1.0,w/1400))
    thick=2
    (tw,th),_=cv2.getTextSize(text,font,scale,thick)
    cv2.putText(img,text,(max(10,(w-tw)//2), int((banner_h+th)/2)-4),
                font,scale,(255,255,255),thick,cv2.LINE_AA)

def draw_dash(img, state, counts, idx):
    h,w=img.shape[:2]; x1,y1=12, max(52, int(h*0.085)); x2,y2=min(w-12,520), y1+105
    overlay=img.copy(); cv2.rectangle(overlay,(x1,y1),(x2,y2),(255,255,255),-1)
    cv2.addWeighted(overlay,.76,img,.24,0,img)
    cv2.rectangle(img,(x1,y1),(x2,y2),STATES[state][3],2)
    font=cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img,"AI School Safety Monitor",(x1+10,y1+25),font,.52,(7,46,116),1,cv2.LINE_AA)
    cv2.putText(img,f"State: {state}",(x1+10,y1+50),font,.43,(25,25,25),1,cv2.LINE_AA)
    cv2.putText(img,f"Abnormal/Fight: {counts['Abnormal Behavior Warning']} | Teacher Check: {counts['Teacher Check Required']}",(x1+10,y1+74),font,.40,(60,60,60),1,cv2.LINE_AA)
    cv2.putText(img,f"Fall: {counts['Fall Detected']} | Immediate: {counts['Immediate Check Required']} | Frame: {idx}",(x1+10,y1+98),font,.40,(60,60,60),1,cv2.LINE_AA)

def motion_region(prev, gray, sens):
    diff=cv2.absdiff(prev,gray)
    diff=cv2.GaussianBlur(diff,(7,7),0)
    _,thr=cv2.threshold(diff,sens,255,cv2.THRESH_BINARY)
    k=np.ones((7,7),np.uint8)
    thr=cv2.morphologyEx(thr,cv2.MORPH_CLOSE,k)
    thr=cv2.dilate(thr,k,iterations=2)
    contours,_=cv2.findContours(thr,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None,0.0,0
    # Use all contours above area for fighting/abnormal crowded motion.
    valid=[c for c in contours if cv2.contourArea(c)>500]
    if not valid: return None,0.0,0
    # combined bbox
    xs=[]; ys=[]; xe=[]; ye=[]; total_area=0
    for c in valid:
        x,y,w,h=cv2.boundingRect(c)
        xs.append(x); ys.append(y); xe.append(x+w); ye.append(y+h)
        total_area += cv2.contourArea(c)
    H,W=gray.shape[:2]
    ratio=total_area/max(1,H*W)
    return (min(xs),min(ys),max(xe),max(ye)),ratio,len(valid)

def decide(bbox, ratio, num_regions, shape, abnormal_th, fight_th, fall_hold, immediate_hold, fall_streak, fight_streak):
    if bbox is None:
        return "Normal Activity",0,max(0,fight_streak-1)
    h,w=shape[:2]; x1,y1,x2,y2=bbox
    bw=max(1,x2-x1); bh=max(1,y2-y1)
    aspect=bw/bh
    center_y=(y1+y2)/2/h

    # Demo rules:
    # Fall: wide and low moving region persists.
    fall_like = aspect > 1.35 and center_y > .52

    # Fighting / abnormal: strong motion, several moving blobs, or very large motion region.
    fighting_like = (ratio > fight_th) or (num_regions >= 2 and ratio > abnormal_th)

    fall_streak = fall_streak + 1 if fall_like else max(0, fall_streak-1)
    fight_streak = fight_streak + 1 if fighting_like else max(0, fight_streak-1)

    if fall_streak >= immediate_hold:
        return "Immediate Check Required", fall_streak, fight_streak
    if fall_streak >= fall_hold:
        return "Fall Detected", fall_streak, fight_streak
    if fight_streak >= 12:
        return "Teacher Check Required", fall_streak, fight_streak
    if fighting_like:
        return "Abnormal Behavior Warning", fall_streak, fight_streak
    return "Normal Activity", fall_streak, fight_streak

def process_video(inp, outp, sens, abnormal_th, fight_th, fall_hold, immediate_hold):
    cap=cv2.VideoCapture(inp)
    if not cap.isOpened(): raise RuntimeError("Không mở được video đầu vào.")
    fps=cap.get(cv2.CAP_PROP_FPS) or 25
    W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280); H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    writer=cv2.VideoWriter(outp,cv2.VideoWriter_fourcc(*"mp4v"),fps,(W,H))
    ok,first=cap.read()
    if not ok: raise RuntimeError("Video không có frame hợp lệ.")
    prev=cv2.cvtColor(first,cv2.COLOR_BGR2GRAY)
    counts={s:0 for s in STATES}; counts["total"]=0
    fall_streak=0; fight_streak=0; idx=0
    bar=st.progress(0); status=st.empty()
    while True:
        ok,frame=cap.read()
        if not ok: break
        idx+=1
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        bbox,ratio,num_regions=motion_region(prev,gray,sens)
        state,fall_streak,fight_streak=decide(bbox,ratio,num_regions,frame.shape,abnormal_th,fight_th,fall_hold,immediate_hold,fall_streak,fight_streak)
        counts[state]+=1; counts["total"]=idx

        # Draw detection
        if bbox:
            x1,y1,x2,y2=bbox
            # red box for abnormal, teacher, fall, immediate
            color=STATES[state][3] if state!="Normal Activity" else STATES["Normal Activity"][3]
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,3 if state!="Normal Activity" else 2)
            draw_label(frame,state,x1,y1-8)

        # Always red banner for warnings.
        draw_big_red_warning(frame,state)

        cv2.putText(frame,f"Motion: {ratio:.3f} | Regions: {num_regions}",(18,H-22),
                    cv2.FONT_HERSHEY_SIMPLEX,.45,(40,40,40),1,cv2.LINE_AA)
        draw_dash(frame,state,counts,idx)
        writer.write(frame); prev=gray
        if total: bar.progress(min(idx/total,1))
        if idx%25==0: status.info(f"Đang xử lý frame {idx}/{total}")
    cap.release(); writer.release()
    bar.progress(1); status.success("Xử lý video hoàn tất.")
    return counts

def render_status(counts):
    cards=""
    for name,(short,vi,hexcolor,_,icon) in STATES.items():
        count=counts.get(name,0)
        cards += f"""<div class="state" style="border-left:6px solid {hexcolor}">
        <div style="font-size:25px">{icon}</div><div><div class="sname">{name}</div><div class="svi">{vi}</div></div>
        <div class="scount" style="color:{hexcolor}">{count:02d}</div></div>"""
    st.markdown(f'<div class="panel"><div class="ptitle">Trạng thái</div><div class="pbody">{cards}</div></div>', unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="AI School Safety Monitor", page_icon="🛡️", layout="wide")
    css()
    if "counts" not in st.session_state:
        st.session_state.counts={s:0 for s in STATES}; st.session_state.counts["total"]=0
    if "output" not in st.session_state: st.session_state.output=None

    st.markdown('<div class="title">Minh họa màn hình demo phần mềm AI School Safety Monitor</div><div class="decor">──── • ─ • ────</div>', unsafe_allow_html=True)
    st.markdown('<div class="shell"><div class="topbar"><div class="brand"><div class="shield">AI</div><div>AI School Safety Monitor <span class="pill">DEMO</span></div></div><div>🔔 <b style="color:red">2</b> &nbsp; ⚙️ &nbsp; 👤 Giáo viên ▾</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="grid">', unsafe_allow_html=True)
    left,mid,right=st.columns([.8,3.25,1.1],gap="medium")
    with left:
        st.markdown('<div class="nav"><div class="navitem active">📹 Giám sát trực tiếp</div><div class="navitem">📋 Sự kiện</div><div class="navitem">🔔 Cảnh báo</div><div class="navitem">🕘 Lịch sử video</div><div class="navitem">📑 Báo cáo</div><div class="navitem">🖥️ Thiết bị</div><div class="navitem">⚙️ Cài đặt</div><hr><div class="navitem">❔ Trợ giúp</div></div>', unsafe_allow_html=True)
    with mid:
        st.markdown('<div class="card"><div class="cardhead"><span>📹 Video sân trường - Demo</span><span>🟢 Trực tuyến ⛶</span></div><div class="cardbody"><div class="upload"><b>📤 Tải video demo lên</b><br><span style="color:#6B7280">Phát hiện: đánh nhau/bất thường, té ngã, cảnh báo đỏ trên video</span></div>', unsafe_allow_html=True)
        up=st.file_uploader("Chọn video",type=["mp4","avi","mov","mkv"],label_visibility="collapsed")
        if up:
            inp=save_uploaded_file(up); st.video(inp)
            with st.expander("⚙️ Cấu hình xử lý", expanded=False):
                sens=st.slider("Độ nhạy phát hiện chuyển động",10,60,28,1)
                abnormal_th=st.slider("Ngưỡng Abnormal Behavior Warning",.005,.120,.025,.005)
                fight_th=st.slider("Ngưỡng đánh nhau / chuyển động mạnh",.010,.200,.050,.005)
                fall_hold=st.slider("Số frame để xác nhận Fall Detected",1,15,3,1)
                immediate_hold=st.slider("Số frame để xác nhận Immediate Check Required",5,60,15,1)
        else:
            inp=None; sens=28; abnormal_th=.025; fight_th=.050; fall_hold=3; immediate_hold=15
            st.info("Vui lòng tải video lên để bắt đầu demo.")
        st.markdown('</div><div class="control"><span>▶</span><span class="live">LIVE</span><span>🔊</span><span>📷</span><span>⬇️</span><span style="margin-left:auto">▦ ▭ ⛶</span></div></div>', unsafe_allow_html=True)
        if inp and st.button("🚀 Bắt đầu xử lý video", type="primary"):
            out=os.path.join(tempfile.gettempdir(),f"AI_School_Safety_Result_{int(time.time())}.mp4")
            try:
                st.session_state.counts=process_video(inp,out,sens,abnormal_th,fight_th,fall_hold,immediate_hold)
                st.session_state.output=out
                st.success("Đã xử lý xong video.")
            except Exception as e: st.error(f"Lỗi: {e}")
        if st.session_state.output and os.path.exists(st.session_state.output):
            st.markdown("### ✅ Video sau xử lý"); st.video(st.session_state.output)
            with open(st.session_state.output,"rb") as f:
                st.download_button("⬇️ Tải video kết quả",f,"AI_School_Safety_Result.mp4","video/mp4")
    with right:
        render_status(st.session_state.counts)
    st.markdown('</div></div><div class="note"><b>AI School Safety Monitor V4:</b> phần mềm demo phát hiện đánh nhau/hành vi bất thường, té ngã và hiển thị cảnh báo màu đỏ trực tiếp trên video. Lưu ý: bản demo dùng xử lý chuyển động OpenCV; triển khai thật nên huấn luyện thêm mô hình AI chuyên biệt.</div>', unsafe_allow_html=True)

if __name__=="__main__":
    main()
