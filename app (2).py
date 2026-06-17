import streamlit as st
import torch
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import tempfile
import io
import time

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SafetyWatch — PPE Detection",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Import fonts */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* App background */
  .stApp { background-color: #0F1117; color: #E8EAED; }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background-color: #161B22;
    border-right: 1px solid #21262D;
  }

  /* Hero banner */
  .hero-banner {
    background: linear-gradient(135deg, #1A2332 0%, #0F1117 50%, #1A1F2E 100%);
    border: 1px solid #21262D;
    border-radius: 12px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
  }
  .hero-banner::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #F59E0B, #EF4444, #3B82F6);
  }
  .hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #F8FAFC;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.02em;
  }
  .hero-sub {
    font-size: 0.95rem;
    color: #8B949E;
    margin: 0;
  }

  /* Metric cards */
  .metric-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; }
  .metric-card {
    flex: 1;
    background: #161B22;
    border: 1px solid #21262D;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    text-align: center;
  }
  .metric-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 0.3rem;
  }
  .metric-label { font-size: 0.78rem; color: #8B949E; text-transform: uppercase; letter-spacing: 0.06em; }
  .metric-safe   .metric-value { color: #22C55E; }
  .metric-danger .metric-value { color: #EF4444; }
  .metric-info   .metric-value { color: #3B82F6; }
  .metric-warn   .metric-value { color: #F59E0B; }

  /* Status banner */
  .status-safe {
    background: linear-gradient(135deg, #052e16, #14532d);
    border: 1px solid #16a34a;
    border-radius: 10px;
    padding: 1rem 1.4rem;
    color: #86EFAC;
    font-weight: 600;
    font-size: 1rem;
    margin-bottom: 1rem;
  }
  .status-danger {
    background: linear-gradient(135deg, #2d0707, #450a0a);
    border: 1px solid #dc2626;
    border-radius: 10px;
    padding: 1rem 1.4rem;
    color: #FCA5A5;
    font-weight: 600;
    font-size: 1rem;
    margin-bottom: 1rem;
  }

  /* Upload zone */
  .upload-zone {
    border: 2px dashed #30363D;
    border-radius: 12px;
    padding: 2.5rem;
    text-align: center;
    background: #161B22;
    margin-bottom: 1rem;
    transition: border-color 0.2s;
  }

  /* Detection badge */
  .badge {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 0.15rem;
  }
  .badge-helmet { background: #052e16; color: #22C55E; border: 1px solid #16a34a; }
  .badge-head   { background: #2d0707; color: #EF4444; border: 1px solid #dc2626; }
  .badge-person { background: #0c1a2e; color: #3B82F6; border: 1px solid #2563eb; }

  /* Confidence bar */
  .conf-bar-bg {
    background: #21262D;
    border-radius: 4px;
    height: 6px;
    margin-top: 4px;
  }

  /* Section headers */
  .section-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #8B949E;
    margin-bottom: 0.6rem;
  }

  /* Streamlit overrides */
  div[data-testid="stFileUploader"] label { color: #8B949E !important; }
  .stSlider > div > div { background: #F59E0B !important; }
  h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; color: #F8FAFC !important; }
  .stButton > button {
    background: #F59E0B !important;
    color: #0F1117 !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    width: 100%;
  }
  .stButton > button:hover { background: #D97706 !important; }
</style>
""", unsafe_allow_html=True)


# ── Model loading ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(model_path: str):
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        return model, None
    except Exception as e:
        return None, str(e)


# ── Inference ────────────────────────────────────────────────────────────────
CLASSES    = ['helmet', 'head', 'person']
BOX_COLORS = {
    'helmet': (34,  197, 94),   # green
    'head'  : (239, 68,  68),   # red
    'person': (59,  130, 246),  # blue
}

def run_inference(model, img_array, conf_thresh, iou_thresh):
    """Run YOLO inference and return annotated image + detections."""
    t0     = time.time()
    result = model.predict(img_array, conf=conf_thresh,
                           iou=iou_thresh, verbose=False)[0]
    ms     = (time.time() - t0) * 1000

    annotated = img_array.copy()
    detections = []

    for box in result.boxes:
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].cpu().tolist()]
        cid   = int(box.cls.cpu())
        score = float(box.conf.cpu())
        cls   = CLASSES[cid] if cid < len(CLASSES) else f'cls{cid}'
        color = BOX_COLORS.get(cls, (255, 255, 255))

        # Draw box
        cv2.rectangle(annotated, (x1,y1), (x2,y2), color, 2)

        # Label background
        label   = f'{cls} {score:.2f}'
        (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, max(y1-th-8,0)),
                      (x1+tw+6, y1), color, -1)
        cv2.putText(annotated, label, (x1+3, max(y1-4,0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

        detections.append({'class': cls, 'conf': score,
                           'box': [x1,y1,x2,y2]})

    return annotated, detections, ms


def compliance_summary(detections):
    """Compute safety compliance metrics."""
    n_helmet = sum(1 for d in detections if d['class'] == 'helmet')
    n_head   = sum(1 for d in detections if d['class'] == 'head')
    n_person = sum(1 for d in detections if d['class'] == 'person')
    total    = n_helmet + n_head
    pct      = int(n_helmet / total * 100) if total > 0 else 0
    return n_helmet, n_head, n_person, pct


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")

    st.markdown('<p class="section-label">Model</p>', unsafe_allow_html=True)
    model_path = st.text_input(
        "Weights path",
        value="best.pt",
        help="Path to your trained best.pt file"
    )

    st.markdown("---")
    st.markdown('<p class="section-label">Detection Settings</p>',
                unsafe_allow_html=True)
    conf_thresh = st.slider("Confidence threshold", 0.1, 0.9, 0.25, 0.05)
    iou_thresh  = st.slider("NMS IoU threshold",    0.1, 0.9, 0.45, 0.05)

    st.markdown("---")
    st.markdown('<p class="section-label">Model Info</p>',
                unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.82rem; color:#8B949E; line-height:1.7">
    <b style="color:#F8FAFC">Architecture:</b> YOLOv8n<br>
    <b style="color:#F8FAFC">Training:</b> Two-phase transfer learning<br>
    <b style="color:#F8FAFC">mAP@50:</b> 64.3%<br>
    <b style="color:#F8FAFC">Helmet AP:</b> 94.5%<br>
    <b style="color:#F8FAFC">Head AP:</b> 96.4%<br>
    <b style="color:#F8FAFC">Dataset:</b> Hard Hat Workers<br>
    <b style="color:#F8FAFC">Classes:</b> helmet · head · person
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p class="section-label">Legend</p>', unsafe_allow_html=True)
    st.markdown("""
    <span class="badge badge-helmet">🟢 helmet</span> — PPE compliant<br><br>
    <span class="badge badge-head">🔴 head</span> — No helmet (violation)<br><br>
    <span class="badge badge-person">🔵 person</span> — Full body
    """, unsafe_allow_html=True)


# ── Main content ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <h1 class="hero-title">🦺 SafetyWatch — PPE Detection</h1>
  <p class="hero-sub">
    Construction site safety monitoring · Helmet & bare-head detection ·
    Two-phase YOLO transfer learning
  </p>
</div>
""", unsafe_allow_html=True)

# Load model
model, model_err = load_model(model_path)

if model_err:
    st.error(f"⚠️ Could not load model from `{model_path}`: {model_err}")
    st.info("Make sure `best.pt` is in the same folder as `app.py`, "
            "or update the path in the sidebar.")
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📸 Image Detection", "🎥 Video Detection", "📊 Model Insights"])

# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    col_up, col_res = st.columns([1, 1.4], gap="large")

    with col_up:
        st.markdown('<p class="section-label">Upload Image</p>',
                    unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Drop a construction site image here",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )

        if uploaded:
            file_bytes = np.frombuffer(uploaded.read(), np.uint8)
            img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            img_rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            st.image(img_rgb, caption="Original image", use_container_width=True)

            if st.button("🔍 Run Detection"):
                with st.spinner("Detecting..."):
                    ann_rgb, detections, ms = run_inference(
                        model, img_rgb, conf_thresh, iou_thresh
                    )

                st.session_state['ann_rgb']    = ann_rgb
                st.session_state['detections'] = detections
                st.session_state['ms']         = ms

    with col_res:
        if 'ann_rgb' in st.session_state:
            ann_rgb    = st.session_state['ann_rgb']
            detections = st.session_state['detections']
            ms         = st.session_state['ms']

            n_helmet, n_head, n_person, pct = compliance_summary(detections)
            total_critical = n_helmet + n_head

            # Compliance status
            if total_critical == 0:
                st.markdown('<div class="status-safe">✅ No heads detected — clear area</div>',
                            unsafe_allow_html=True)
            elif n_head == 0:
                st.markdown(f'<div class="status-safe">✅ Full compliance — all {n_helmet} workers wearing helmets</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div class="status-danger">⚠️ Safety violation — '
                    f'{n_head} worker{"s" if n_head>1 else ""} without helmet detected</div>',
                    unsafe_allow_html=True
                )

            # Metric cards
            st.markdown(f"""
            <div class="metric-row">
              <div class="metric-card metric-safe">
                <div class="metric-value">{n_helmet}</div>
                <div class="metric-label">With Helmet</div>
              </div>
              <div class="metric-card metric-danger">
                <div class="metric-value">{n_head}</div>
                <div class="metric-label">No Helmet</div>
              </div>
              <div class="metric-card metric-warn">
                <div class="metric-value">{pct}%</div>
                <div class="metric-label">Compliance</div>
              </div>
              <div class="metric-card metric-info">
                <div class="metric-value">{ms:.0f}ms</div>
                <div class="metric-label">Inference</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Annotated image
            st.markdown('<p class="section-label">Detection Result</p>',
                        unsafe_allow_html=True)
            st.image(ann_rgb, use_container_width=True)

            # Download button
            pil_img    = Image.fromarray(ann_rgb)
            buf        = io.BytesIO()
            pil_img.save(buf, format='JPEG', quality=92)
            st.download_button(
                "⬇️ Download annotated image",
                data=buf.getvalue(),
                file_name="safetywatch_result.jpg",
                mime="image/jpeg",
            )

            # Detection list
            if detections:
                st.markdown('<p class="section-label" style="margin-top:1rem">Detections</p>',
                            unsafe_allow_html=True)
                for i, d in enumerate(detections):
                    cls   = d['class']
                    badge = f'badge-{cls}'
                    conf  = d['conf']
                    bar_w = int(conf * 100)
                    bar_c = {'helmet':'#22C55E','head':'#EF4444','person':'#3B82F6'}.get(cls,'#888')
                    st.markdown(f"""
                    <div style="display:flex;align-items:center;gap:0.8rem;
                                padding:0.5rem 0;border-bottom:1px solid #21262D">
                      <span class="badge {badge}">{cls}</span>
                      <div style="flex:1">
                        <div style="font-size:0.78rem;color:#8B949E">conf {conf:.3f}</div>
                        <div class="conf-bar-bg">
                          <div style="width:{bar_w}%;background:{bar_c};height:6px;border-radius:4px"></div>
                        </div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="upload-zone">
              <div style="font-size:2.5rem;margin-bottom:0.8rem">📸</div>
              <div style="color:#F8FAFC;font-weight:600;margin-bottom:0.4rem">
                Upload an image to get started
              </div>
              <div style="color:#8B949E;font-size:0.88rem">
                Construction site photos work best
              </div>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<p class="section-label">Video / Webcam</p>',
                unsafe_allow_html=True)

    video_file = st.file_uploader(
        "Upload a video file",
        type=["mp4", "avi", "mov"],
        label_visibility="collapsed"
    )

    if video_file:
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            tmp.write(video_file.read())
            tmp_path = tmp.name

        st.video(tmp_path)

        if st.button("🎬 Process Video"):
            cap         = cv2.VideoCapture(tmp_path)
            total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps         = cap.get(cv2.CAP_PROP_FPS) or 25
            w           = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h           = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            out_path    = tmp_path.replace('.mp4', '_annotated.mp4')
            fourcc      = cv2.VideoWriter_fourcc(*'mp4v')
            writer      = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

            progress    = st.progress(0)
            status_txt  = st.empty()
            preview_ph  = st.empty()

            frame_idx   = 0
            all_summary = {'helmet':0,'head':0,'person':0}

            while True:
                ret, frame = cap.read()
                if not ret: break

                frame_rgb        = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ann_rgb, dets, _ = run_inference(model, frame_rgb,
                                                 conf_thresh, iou_thresh)
                ann_bgr          = cv2.cvtColor(ann_rgb, cv2.COLOR_RGB2BGR)
                writer.write(ann_bgr)

                for d in dets:
                    all_summary[d['class']] = all_summary.get(d['class'],0)+1

                frame_idx += 1
                pct = int(frame_idx / max(total_frame,1) * 100)
                progress.progress(pct)

                if frame_idx % 10 == 0:
                    preview_ph.image(ann_rgb, caption=f"Frame {frame_idx}/{total_frame}",
                                     use_container_width=True)
                    status_txt.text(f"Processing frame {frame_idx}/{total_frame}…")

            cap.release(); writer.release()
            progress.progress(100)
            status_txt.text("✅ Done!")

            with open(out_path,'rb') as f:
                st.download_button("⬇️ Download annotated video",
                                   data=f.read(), file_name="safetywatch_video.mp4",
                                   mime="video/mp4")

            nh = all_summary.get('helmet',0)
            nd = all_summary.get('head',0)
            st.markdown(f"""
            <div class="metric-row" style="margin-top:1rem">
              <div class="metric-card metric-safe">
                <div class="metric-value">{nh}</div>
                <div class="metric-label">Helmet detections</div>
              </div>
              <div class="metric-card metric-danger">
                <div class="metric-value">{nd}</div>
                <div class="metric-label">Violation detections</div>
              </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Upload an MP4/AVI/MOV video file to process frame-by-frame detection.")

# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### Model Performance")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div style="background:#161B22;border:1px solid #21262D;border-radius:10px;padding:1.4rem">
          <div class="section-label">Per-Class AP@50</div>
          <div style="margin-top:0.8rem">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.6rem">
              <span class="badge badge-helmet">helmet</span>
              <span style="font-family:'Space Grotesk',sans-serif;font-weight:700;color:#22C55E">94.5%</span>
            </div>
            <div style="background:#21262D;border-radius:4px;height:8px;margin-bottom:1rem">
              <div style="width:94.5%;background:#22C55E;height:8px;border-radius:4px"></div>
            </div>
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.6rem">
              <span class="badge badge-head">head (no helmet)</span>
              <span style="font-family:'Space Grotesk',sans-serif;font-weight:700;color:#EF4444">96.4%</span>
            </div>
            <div style="background:#21262D;border-radius:4px;height:8px;margin-bottom:1rem">
              <div style="width:96.4%;background:#EF4444;height:8px;border-radius:4px"></div>
            </div>
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.6rem">
              <span class="badge badge-person">person</span>
              <span style="font-family:'Space Grotesk',sans-serif;font-weight:700;color:#3B82F6">2.0%</span>
            </div>
            <div style="background:#21262D;border-radius:4px;height:8px">
              <div style="width:2%;background:#3B82F6;height:8px;border-radius:4px"></div>
            </div>
            <div style="font-size:0.75rem;color:#8B949E;margin-top:0.5rem">
              ↑ Low person AP due to 27.4× class imbalance in training data
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div style="background:#161B22;border:1px solid #21262D;border-radius:10px;padding:1.4rem">
          <div class="section-label">Overall Metrics</div>
          <div style="margin-top:0.8rem">
            <table style="width:100%;font-size:0.88rem;border-collapse:collapse">
              <tr style="border-bottom:1px solid #21262D">
                <td style="padding:0.5rem 0;color:#8B949E">mAP@50</td>
                <td style="text-align:right;font-weight:700;color:#F59E0B;font-family:'Space Grotesk',sans-serif">64.3%</td>
              </tr>
              <tr style="border-bottom:1px solid #21262D">
                <td style="padding:0.5rem 0;color:#8B949E">mAP@50-95</td>
                <td style="text-align:right;font-weight:700;color:#F8FAFC;font-family:'Space Grotesk',sans-serif">42.6%</td>
              </tr>
              <tr style="border-bottom:1px solid #21262D">
                <td style="padding:0.5rem 0;color:#8B949E">Precision</td>
                <td style="text-align:right;font-weight:700;color:#F8FAFC;font-family:'Space Grotesk',sans-serif">62.1%</td>
              </tr>
              <tr style="border-bottom:1px solid #21262D">
                <td style="padding:0.5rem 0;color:#8B949E">Recall</td>
                <td style="text-align:right;font-weight:700;color:#F8FAFC;font-family:'Space Grotesk',sans-serif">61.6%</td>
              </tr>
              <tr style="border-bottom:1px solid #21262D">
                <td style="padding:0.5rem 0;color:#8B949E">Inference speed</td>
                <td style="text-align:right;font-weight:700;color:#F8FAFC;font-family:'Space Grotesk',sans-serif">~4ms / image</td>
              </tr>
              <tr>
                <td style="padding:0.5rem 0;color:#8B949E">Parameters</td>
                <td style="text-align:right;font-weight:700;color:#F8FAFC;font-family:'Space Grotesk',sans-serif">3.2M</td>
              </tr>
            </table>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Training Strategy")
    st.markdown("""
    <div style="background:#161B22;border:1px solid #21262D;border-radius:10px;
                padding:1.4rem;margin-top:0.5rem">
      <div style="display:flex;gap:2rem;flex-wrap:wrap">
        <div style="flex:1;min-width:200px">
          <div class="section-label">Phase 1 — Warm-up (15 epochs)</div>
          <div style="font-size:0.88rem;color:#E8EAED;line-height:1.7">
            First 9 backbone layers frozen.<br>
            Only neck + detection head trained.<br>
            LR: <code style="color:#F59E0B">1e-3</code>
          </div>
        </div>
        <div style="flex:1;min-width:200px">
          <div class="section-label">Phase 2 — Fine-tuning (50 epochs)</div>
          <div style="font-size:0.88rem;color:#E8EAED;line-height:1.7">
            All layers unfrozen.<br>
            End-to-end fine-tuning.<br>
            LR: <code style="color:#F59E0B">1e-4</code> (10× lower)
          </div>
        </div>
        <div style="flex:1;min-width:200px">
          <div class="section-label">Dataset</div>
          <div style="font-size:0.88rem;color:#E8EAED;line-height:1.7">
            Hard Hat Workers (Roboflow)<br>
            7,000+ images · 3 classes<br>
            Construction site domain
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

