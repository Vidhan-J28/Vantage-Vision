import streamlit as st
import sys 
from pathlib import Path
import time
from src.inference import YOLOv11Inference
from src.utils import save, load_data, get_class_freq
from src.config import load_config
from PIL import Image, ImageDraw, ImageFont, ImageOps
import base64
import io
import json

try:
    APP_CONFIG = load_config()
except FileNotFoundError:
    APP_CONFIG = {}

MODEL_DEFAULT = APP_CONFIG.get("model", {}).get("yolo", "yolo11m.pt")
GRID_DEFAULT = APP_CONFIG.get("app", {}).get("default_grid", 4)
SHOW_BOXES_DEFAULT = APP_CONFIG.get("app", {}).get("show_boxes", False)
HIGHLIGHT_DEFAULT = APP_CONFIG.get("app", {}).get("highlight_matches", True)
SEARCH_MODE_DEFAULT = APP_CONFIG.get("app", {}).get("search_mode", "Atleast one of the selected ones")
CONF_THRESHOLD = APP_CONFIG.get("model", {}).get("conf_threshold", 0.3)



@st.cache_data
def load_image(path):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)  # normalize orientation NOW, upfront.
                                         # cv2.imread on this OpenCV build already auto-rotates
                                         # to this same corrected orientation before YOLO sees it,
                                         # so this keeps the display frame and detection frame in sync.
    original_size = img.size            # (width, height) in that corrected/upright frame
    img.thumbnail((900, 900))           # Resize once, already upright
    return img.copy(), original_size

@st.cache_resource
def get_font():
    try:
        return ImageFont.truetype("arial.ttf",18)
    except:
        return ImageFont.load_default(size=18)

sys.path.append(str(Path(__file__).parent))

@st.cache_data
def img_to_base64(image_bytes):

    img=Image.open(io.BytesIO(image_bytes))

    buffered=io.BytesIO()

    img.save(buffered,format="PNG")

    return base64.b64encode(buffered.getvalue()).decode()

def init_session_state():
    session_defaults={
        'metadata':None,
        'unique_classes'  :[None],
        'freq_classes':{None},
        'search_param' : {
            'mode' : SEARCH_MODE_DEFAULT,
            'selected_ones' : [],
            'maxcnt' : {} #max cnt of each selected class by user
        },
        'results' : [],
        'show_boxes' : SHOW_BOXES_DEFAULT,
        'grid' : GRID_DEFAULT,
        'highlight_matches' : HIGHLIGHT_DEFAULT,
        
        
    }
    
    for key,value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key]=value
            
init_session_state()

st.set_page_config(page_title='Lumen Vision' , page_icon='✨' , layout='wide')
st.title("✨ Vantage Vision")
st.caption("AI-powered visual search, refined.")
st.markdown("""
<div class="how-it-works">
    <div class="step-card">
        <div class="step-num">1</div>
        <div class="step-title">Analyze</div>
        <div class="step-text">Point Lumen Vision at a folder of images. A YOLOv11 model scans every photo and detects the objects in it, no manual tagging needed.</div>
    </div>
    <div class="step-card">
        <div class="step-num">2</div>
        <div class="step-title">Store</div>
        <div class="step-text">Detections are saved to a metadata file you can reload anytime, so you only need to analyze a folder once.</div>
    </div>
    <div class="step-card">
        <div class="step-num">3</div>
        <div class="step-title">Search</div>
        <div class="step-text">Pick the classes you care about, choose "at least one" or "all", and instantly filter down to the matching images.</div>
    </div>
    <div class="step-card">
        <div class="step-num">4</div>
        <div class="step-title">Review</div>
        <div class="step-text">Browse results in a resizable grid with optional bounding boxes, and highlight only the classes you searched for.</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Custom CSS - Glassmorphism theme, green/teal accent
st.markdown(f"""
<style>

/* ---------- Global background (darker slate palette, static) ---------- */
.stApp {{
    background: radial-gradient(circle at 15% 15%, #080c10 0%, #030507 45%, #000000 100%);
    background-attachment: fixed;
}}

/* ---------- Pulsing glow layer ---------- */
.glow-layer {{
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    overflow: hidden;
    pointer-events: none;
    z-index: 0;
}}

.glow-blob {{
    position: absolute;
    border-radius: 50%;
    filter: blur(110px);
    animation: pulseGlow 9s ease-in-out infinite;
}}

.glow-teal {{
    top: 8%;
    left: 8%;
    width: 560px;
    height: 560px;
    background: rgba(99, 102, 241, 0.45);
}}

.glow-amber {{
    bottom: 6%;
    right: 10%;
    width: 500px;
    height: 500px;
    background: rgba(217, 70, 239, 0.32);
    animation-delay: 3.5s;
}}

@keyframes pulseGlow {{
    0%, 100% {{ opacity: 0.55; transform: scale(1); }}
    50%      {{ opacity: 1; transform: scale(1.3); }}
}}

/* ---------- Floating particles layer ---------- */
.particles-layer {{
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    overflow: hidden;
    pointer-events: none;
    z-index: 0;
}}

.particle {{
    position: absolute;
    bottom: -40px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(99, 102, 241, 0.55), transparent 70%);
    animation: floatUp linear infinite;
}}

.particle.amber {{
    background: radial-gradient(circle, rgba(217, 70, 239, 0.5), transparent 70%);
}}

@keyframes floatUp {{
    0%   {{ transform: translateY(0) translateX(0); opacity: 0; }}
    10%  {{ opacity: 0.7; }}
    90%  {{ opacity: 0.5; }}
    100% {{ transform: translateY(-115vh) translateX(30px); opacity: 0; }}
}}

/* Keep actual app content above the particle layer */
.main .block-container {{
    position: relative;
    z-index: 1;
}}

/* Base text color + size inside main app */
.stApp, .stApp p, .stApp span, .stApp label, .stMarkdown {{
    color: #e5e7eb;
    font-size: 17px;
}}

div[data-testid="stCaptionContainer"] {{
    font-size: 16px !important;
}}

/* Title */
h1, h1 span {{
    background: linear-gradient(90deg, #14b8a6, #64748b, #8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 800;
    font-size: 2.75rem !important;
    letter-spacing: -0.5px;
    padding-bottom: 4px;
}}

h2, h3, h4 {{
    color: #818cf8 !important;
}}

/* Main container adjustments */
.st-emotion-cache-1v0mbdj {{
    width: 100% !important;
    height: 100% !important;
}}

/* Column container - critical for grid layout */
.st-emotion-cache-1wrcr25 {{
    max-width: none !important;
    padding: 0 1rem !important;
}}

/* Individual column styling */
.st-emotion-cache-1n76uvr {{
    padding: 0.5rem !important;
}}

/* ---------- Glass panels: expanders / containers ---------- */
div[data-testid="stExpander"] {{
    background: rgba(255, 255, 255, 0.04);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(100, 116, 139, 0.35);
    border-radius: 16px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
    margin-bottom: 1rem;
}}

div[data-testid="stExpander"] summary {{
    color: #a78bfa !important;
    font-weight: 600;
}}

/* ---------- Text inputs ---------- */
div[data-testid="stTextInput"] input {{
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(100, 116, 139, 0.45) !important;
    border-radius: 10px !important;
    color: #e5e7eb !important;
    backdrop-filter: blur(6px);
}}

div[data-testid="stTextInput"] input:focus {{
    border: 1px solid #6366f1 !important;
    box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2) !important;
}}

/* ---------- Buttons ---------- */
.stButton > button, .stDownloadButton > button {{
    background: linear-gradient(135deg, #4338ca, #6d28d9);
    color: #e5e7eb !important;
    border: 1px solid rgba(139, 92, 246, 0.35);
    border-radius: 12px;
    padding: 0.55rem 1.4rem;
    font-weight: 600;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
    transition: all 0.2s ease;
}}

.stButton > button:hover, .stDownloadButton > button:hover {{
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    background: linear-gradient(135deg, #6366f1, #a21caf);
    border: 1px solid rgba(217, 70, 239, 0.6);
    color: #e5e7eb !important;
}}

.stButton > button:active, .stDownloadButton > button:active {{
    transform: translateY(0px);
}}

/* ---------- Radio buttons ---------- */
div[data-testid="stRadio"] > div {{
    background: rgba(255, 255, 255, 0.03);
    border-radius: 12px;
    padding: 0.5rem 0.8rem;
    border: 1px solid rgba(100, 116, 139, 0.3);
}}

div[data-testid="stRadio"] label {{
    color: #e5e7eb !important;
}}

/* ---------- Multiselect / Selectbox ---------- */
div[data-testid="stMultiSelect"] > div, div[data-baseweb="select"] > div {{
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(100, 116, 139, 0.45) !important;
    border-radius: 10px !important;
    color: #e5e7eb !important;
}}

span[data-baseweb="tag"] {{
    background: linear-gradient(135deg, #d946ef, #7c3aed) !important;
    border-radius: 8px !important;
}}

/* ---------- Slider ---------- */
div[data-testid="stSlider"] div[role="slider"] {{
    background-color: #d946ef !important;
    box-shadow: 0 0 0 4px rgba(217, 70, 239, 0.18) !important;
}}

div[data-testid="stTickBar"] {{
    background: transparent !important;
}}

/* ---------- Checkbox ---------- */
div[data-testid="stCheckbox"] label span {{
    border-color: #6366f1 !important;
}}

/* ---------- Alerts (success / warning / error) ---------- */
div[data-testid="stAlert"] {{
    border-radius: 12px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(100, 116, 139, 0.3);
}}

/* ---------- Image result cards (glassmorphism) ---------- */
.image-card {{
    border-radius: 18px;
    overflow: hidden;
    background: rgba(255, 255, 255, 0.04);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(100, 116, 139, 0.35);
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.4);
    transition: all 0.3s ease;
    margin-bottom: 20px;
}}

.image-card:hover {{
    transform: translateY(-4px);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5);
    border: 1px solid rgba(139, 92, 246, 0.5);
}}

.image-container {{
    position: relative;
    width: 100%;
    aspect-ratio: 4/3;
    background: rgba(0, 0, 0, 0.35);
}}

.image-container img {{
    width: 100%;
    height: 100%;
    object-fit: contain;
}}

.meta-overlay {{
    padding: 12px 14px;
    background: rgba(10, 14, 19, 0.8);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    color: #e5e7eb;
    font-size: 18px;
    line-height: 1.5;
    border-top: 1px solid rgba(100, 116, 139, 0.3);
}}

.meta-overlay strong {{
    color: #a78bfa;
    font-size: 19px;
}}

/* ---------- How it works strip ---------- */
.how-it-works {{
    display: flex;
    gap: 16px;
    margin: 4px 0 24px 0;
    flex-wrap: wrap;
}}

.step-card {{
    flex: 1;
    min-width: 220px;
    background: rgba(255, 255, 255, 0.04);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(100, 116, 139, 0.35);
    border-radius: 16px;
    padding: 16px 18px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
    transition: all 0.2s ease;
}}

.step-card:hover {{
    border: 1px solid rgba(139, 92, 246, 0.5);
    transform: translateY(-2px);
}}

.step-num {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 26px;
    height: 26px;
    border-radius: 50%;
    background: linear-gradient(135deg, #6366f1, #8b5cf6, #d946ef);
    color: #e5e7eb;
    font-weight: 700;
    font-size: 14px;
    margin-bottom: 8px;
}}

.step-title {{
    color: #a78bfa;
    font-weight: 600;
    font-size: 17px;
    margin-bottom: 4px;
}}

.step-text {{
    color: #9ca3af;
    font-size: 15px;
    line-height: 1.45;
}}
</style>
""", unsafe_allow_html=True)

# Ambient pulsing glow (purely decorative, sits behind all content)
st.markdown("""
<div class="glow-layer">
    <div class="glow-blob glow-teal"></div>
    <div class="glow-blob glow-amber"></div>
</div>
""", unsafe_allow_html=True)

# Ambient floating particles (purely decorative, sits behind all content)
st.markdown("""
<div class="particles-layer">
    <div class="particle"       style="left:4%;  width:10px; height:10px; animation-duration:19s; animation-delay:0s;"></div>
    <div class="particle amber" style="left:12%; width:6px;  height:6px;  animation-duration:14s; animation-delay:2s;"></div>
    <div class="particle"       style="left:20%; width:14px; height:14px; animation-duration:24s; animation-delay:1s;"></div>
    <div class="particle amber" style="left:29%; width:8px;  height:8px;  animation-duration:17s; animation-delay:5s;"></div>
    <div class="particle"       style="left:37%; width:5px;  height:5px;  animation-duration:12s; animation-delay:3s;"></div>
    <div class="particle"       style="left:46%; width:12px; height:12px; animation-duration:21s; animation-delay:6s;"></div>
    <div class="particle amber" style="left:55%; width:9px;  height:9px;  animation-duration:16s; animation-delay:0.5s;"></div>
    <div class="particle"       style="left:63%; width:7px;  height:7px;  animation-duration:18s; animation-delay:4s;"></div>
    <div class="particle"       style="left:71%; width:15px; height:15px; animation-duration:23s; animation-delay:2.5s;"></div>
    <div class="particle amber" style="left:79%; width:6px;  height:6px;  animation-duration:13s; animation-delay:7s;"></div>
    <div class="particle"       style="left:87%; width:11px; height:11px; animation-duration:20s; animation-delay:1.5s;"></div>
    <div class="particle"       style="left:94%; width:8px;  height:8px;  animation-duration:15s; animation-delay:3.5s;"></div>
</div>
""", unsafe_allow_html=True)


# Main features

option = st.radio("Choose one : " , ("Analyze New Images" , "Load Existing Data") , horizontal=True)

if(option=="Analyze New Images"):
    with st.expander("Analyze New Images" , expanded=True):
        col1,col2 = st.columns(2)
        with col1:
            img_dir = st.text_input("Enter Image Directory Path" , placeholder="path//for//images")
        with col2:
            model_path = st.text_input("Enter Model Weights Path" , MODEL_DEFAULT)

        if(st.button("Start Inferencing")):
            if img_dir:
                try:
                    with st.spinner("Detecting Objects...."):
                        infer = YOLOv11Inference(model_path)
                        all_data = infer.analyze_dir(img_dir)
                        metadata_path = save(all_data , img_dir)
                        st.success(f"Analyzed All Imgaes ({len(all_data)}). The data is stored at :")
                        st.code(str(metadata_path))
                        
                        st.session_state.metadata=all_data 
                        st.session_state.unique_classes , st.session_state.freq_classes = get_class_freq(all_data)
                except Exception as e:
                    st.error(f"Error occured {str(e)}")
                    
            
            else:
                st.warning("Please enter a valid path")
                    

else:
    with st.expander("Load Existing Data" , expanded=True):
        data_path=st.text_input("Data File Path : " , placeholder= "path//for//data.json")
        
        if(st.button("Load Data")):
            
            if(data_path):
                try:
                    with st.spinner("Loading Metadata...."):
                        loaded_data = load_data(data_path)
                        st.session_state.metadata=loaded_data 
                        st.session_state.unique_classes , st.session_state.freq_classes = get_class_freq(loaded_data)
                        
                        st.success(f"Data Loaded Successfully ({len(loaded_data)} images). ")
                except Exception as e:
                    st.error(f"Error occured {str(e)}")
                
            else:
                st.warning("Please enter a valid path")
                

# Search Function

if st.session_state.metadata:
    st.header("Discover")         
    
    with st.container():
        st.session_state.search_param['mode'] = st.radio("Mode Of Search : " , ("Atleast one of the selected ones" , "All of the selected ones") , horizontal=True)
        
        st.session_state.search_param['selected_ones'] = st.multiselect("All Possible Classes :" , options=st.session_state.unique_classes)
        
        if st.session_state.search_param['selected_ones']:
            st.subheader('(Optional) Threshold Count :  ')
            
            cols =st.columns(len(st.session_state.search_param['selected_ones']))
            
            for i,cls in enumerate(st.session_state.search_param['selected_ones']):
                with cols[i]:
                    st.session_state.search_param['maxcnt'][cls] = st.selectbox(f"Maximum {cls} you want" ,options=["None"] + st.session_state.freq_classes[cls])
                    

           
        if st.button("Find Images", type="primary") and st.session_state.search_param['selected_ones']:
            result=[]
            
            for item in st.session_state.metadata:
                matches=False
                class_matches={}
                
                for cls in st.session_state.search_param['selected_ones']:
                    class_detect = [d for d in item['detections'] if d['class']==cls]
                    class__found_count = len(class_detect)
                    class_matches[cls]=False
                    
                    maxcnt=st.session_state.search_param['maxcnt'].get(cls,"None")
                    
                    #setting class_matches for each class to be true whther we should show that image or not wrt limit entered by user
                    
                    if maxcnt=="None":
                        class_matches[cls]=(class__found_count>=1)
                    else:
                        class_matches[cls] = (class__found_count>=1 and class__found_count<=int(maxcnt))
                        
                        
                #using any , all for respective options as if any class is present then true for any 
                #and only if all classes are present then true for all
                        
                if st.session_state.search_param['mode']=="Atleast one of the selected ones":
                    matches = any(class_matches.values())
                else:
                    matches = all(class_matches.values())
                    
                    
                if matches:
                    result.append(item) #image=item from all_data
                    
            st.session_state.results = result
            
#display images

if st.session_state.results:

    results=st.session_state.results

    st.subheader(f"{len(results)} matching images")

    with st.expander("View Options : " , expanded=True):
        cols = st.columns(3)
        with cols[0]:
            st.session_state.show_boxes=st.checkbox("Show Object Detections" , value=st.session_state.show_boxes)
            
        with cols[1]:
            st.session_state.grid=st.slider("Grid size : " , min_value=2, max_value=6 ,value=st.session_state.grid)
            
        with cols[2]:
            st.session_state.highlight_matches=st.checkbox("Highlight Matching Classes" , value=st.session_state.highlight_matches)
        
        
    
    #create grid
    
    grid_cols=st.columns(st.session_state.grid)
    col_index=0
    
    for res in results:
        with grid_cols[col_index]:
            try:
                img, original_size = load_image(res["image_path"])
                img = img.copy().convert("RGBA")

                scale_x = img.width / original_size[0]
                scale_y = img.height / original_size[1]

                draw = ImageDraw.Draw(img)

                if st.session_state.show_boxes:

                    font = get_font()

                    for det in res["detections"]:

                        cls = det["class"]

                        if det["confidence"] < CONF_THRESHOLD:
                            continue

                        raw_bbox = det["coord"]
                        bbox = [
                            raw_bbox[0] * scale_x,
                            raw_bbox[1] * scale_y,
                            raw_bbox[2] * scale_x,
                            raw_bbox[3] * scale_y,
                        ]

                        thickness = 4

                        if cls in st.session_state.search_param["selected_ones"]:
                            color = "#00D9FF"

                        elif not st.session_state.highlight_matches:
                            color = "#FF6B6B"

                        else:
                            continue

                        draw.rounded_rectangle(bbox, radius=10, outline=color, width=thickness)

                        label = f"{cls} {det['confidence']:.2f}"

                        text_box = draw.textbbox((0, 0), label, font=font)

                        text_w = text_box[2] - text_box[0]
                        text_h = text_box[3] - text_box[1]

                        label_patch = Image.new("RGBA", (text_w + 8, text_h + 4), color)
                        label_draw = ImageDraw.Draw(label_patch)
                        label_draw.text((4, 2), label, fill="black", font=font)

                        img.paste(label_patch, (int(bbox[0]), int(bbox[1])), label_patch)

                img = img.convert("RGB")
                    
                data_items = [f"{k} : {v} " for k,v in res['class_counts'].items() if k in st.session_state.search_param['selected_ones']]
                    
                    #image card
                    
                # prepare image bytes as base64
                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
                img_b64 = base64.b64encode(buffer.getvalue()).decode()

                st.markdown(f"""
                    <div class="image-card">
                    <div class="image-container">
                        <img src="data:image/png;base64,{img_b64}">
                    </div>
                    <div class="meta-overlay">
                        <strong>{Path(res['image_path']).name}</strong><br>
                        {", ".join(data_items) if data_items else "No matches"}
                    </div>
                    </div>
                    """, unsafe_allow_html=True)
                        
            except Exception as e:
                st.error(f"Error displaying {res['image_path']} : {str(e)}")
             
        col_index = (col_index+1)% st.session_state.grid   
        
                
                
    with st.expander("Export Options"):
        st.download_button(
            label="Download Results (JSON)",
            data = json.dumps(results,indent=2),
            file_name="search_results.json",
            mime="application/json"
        )

                    
                
            
#C:\Users\ROG\Desktop\tp\Analyzed\tp\ALL_DATA.json