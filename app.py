import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="PlantGuard AI", page_icon="🌿", layout="wide")

# Visual styling only. All visible content uses Streamlit components,
# so HTML tags cannot accidentally appear as text on the page.
st.markdown(r'''<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');
.stApp{background:linear-gradient(135deg,#f4faef,#e8f4e4,#f8fbf6);color:#173d28}.block-container{max-width:1180px;padding-top:2rem;padding-bottom:3rem}html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
.hero{background:linear-gradient(135deg,#123f29,#267044,#5b9554);padding:42px 48px;border-radius:28px;margin-bottom:30px;box-shadow:0 18px 45px rgba(31,91,48,.2)}
.badge{display:inline-block;color:white;background:rgba(255,255,255,.16);border:1px solid rgba(255,255,255,.28);border-radius:30px;padding:7px 14px;font-size:12px;font-weight:700}.hero h1{color:white!important;font-family:'Playfair Display',serif;font-size:48px;margin:14px 0 5px}.hero p{color:#e8f5e7!important;font-size:17px;line-height:1.6;max-width:760px}
.title{color:#174d32!important;font-family:'Playfair Display',serif;font-size:30px;font-weight:700;margin-top:18px}.sub{color:#627565!important;font-size:15px;margin-bottom:18px}
.card{background:white;border:1px solid #d7e7d3;border-radius:20px;padding:22px;box-shadow:0 8px 22px rgba(39,76,47,.06)}.card h3{color:#174d32!important;margin:0 0 8px}.card p{color:#5f7162!important;line-height:1.55}
.upload{background:#fff;border:2px dashed #8abd83;border-radius:24px;padding:22px;box-shadow:0 10px 28px rgba(39,76,47,.08)}.upload h3{color:#174d32!important;margin:0 0 4px}.upload p{color:#6d7f6f!important;margin:0}
[data-testid="stFileUploaderDropzone"]{background:#fbfefa!important;border:1px solid #d5e7d1!important;border-radius:18px!important}[data-testid="stFileUploaderDropzone"] *{color:#385840!important}
.stButton>button{width:100%;border:0;border-radius:14px;background:linear-gradient(135deg,#2d7748,#174d32);color:white!important;font-weight:700;font-size:16px;min-height:50px;box-shadow:0 8px 20px rgba(23,77,50,.18)}.stButton>button:hover{background:linear-gradient(135deg,#3d8b58,#1d603b);color:white!important}
.result-good{background:linear-gradient(135deg,#edf8e9,#fff);border:1px solid #b8d9b2;border-left:7px solid #3f824c;border-radius:22px;padding:24px}.result-review{background:linear-gradient(135deg,#fff8e8,#fff);border:1px solid #e5cd96;border-left:7px solid #d49328;border-radius:22px;padding:24px}.result-title{color:#174d32!important;font-size:25px;font-weight:700}.result-review .result-title{color:#765014!important}.small{color:#687a6b!important;font-size:13px}
.step{background:white;border:1px solid #d7e8d3;border-radius:20px;padding:22px;min-height:145px;box-shadow:0 8px 22px rgba(39,76,47,.06)}.step h3{color:#174d32!important}.step p{color:#627565!important;line-height:1.5}.footer{text-align:center;color:#6d7f6f!important;font-size:13px;padding:25px 0 5px}[data-testid="stMetric"]{background:white;border:1px solid #d7e7d3;border-radius:18px;padding:16px}[data-testid="stProgressBar"]>div>div>div{background:linear-gradient(90deg,#3f824c,#83b85f)}[data-testid="stImage"] img{border-radius:18px}hr{border-color:#d8e7d5}
</style>''', unsafe_allow_html=True)

MODEL_PATH=os.path.join(os.path.dirname(os.path.abspath(__file__)),"plant_disease_model.keras")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model=load_model()

CLASS_NAMES=[
"Apple___Apple_scab","Apple___Black_rot","Apple___Cedar_apple_rust","Apple___healthy",
"Blueberry___healthy","Cherry_(including_sour)___Powdery_mildew","Cherry_(including_sour)___healthy",
"Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot","Corn_(maize)___Common_rust_","Corn_(maize)___Northern_Leaf_Blight","Corn_(maize)___healthy",
"Grape___Black_rot","Grape___Esca_(Black_Measles)","Grape___Leaf_blight_(Isariopsis_Leaf_Spot)","Grape___healthy",
"Orange___Haunglongbing_(Citrus_greening)","Peach___Bacterial_spot","Peach___healthy","Pepper,_bell___Bacterial_spot","Pepper,_bell___healthy",
"Potato___Early_blight","Potato___Late_blight","Potato___healthy","Raspberry___healthy","Soybean___healthy","Squash___Powdery_mildew",
"Strawberry___Leaf_scorch","Strawberry___healthy","Tomato___Bacterial_spot","Tomato___Early_blight","Tomato___Late_blight","Tomato___Leaf_Mold",
"Tomato___Septoria_leaf_spot","Tomato___Spider_mites_Two-spotted_spider_mite","Tomato___Target_Spot","Tomato___Tomato_Yellow_Leaf_Curl_Virus",
"Tomato___Tomato_mosaic_virus","Tomato___healthy"]

DISEASE_INFO={
"Apple___Apple_scab":"A fungal disease that causes dark spots and lesions on apple leaves and fruit.",
"Apple___Black_rot":"A fungal disease that can cause dark leaf spots and fruit rot.",
"Apple___Cedar_apple_rust":"A fungal disease that produces yellow-orange spots on apple leaves.",
"Cherry_(including_sour)___Powdery_mildew":"Powdery mildew appears as white powder-like growth on leaves and shoots.",
"Corn_(maize)___Common_rust_":"Common rust produces reddish-brown rust-like spots on corn leaves.",
"Corn_(maize)___Northern_Leaf_Blight":"Northern leaf blight produces long gray-green or brown lesions on corn leaves.",
"Grape___Black_rot":"A fungal disease that causes brown or black lesions and fruit rot.",
"Potato___Early_blight":"Early blight causes dark circular spots and yellowing on potato leaves.",
"Potato___Late_blight":"Late blight can cause dark lesions and may spread rapidly.",
"Squash___Powdery_mildew":"Powdery mildew produces white powder-like growth on squash leaves.",
"Strawberry___Leaf_scorch":"Leaf scorch causes dark purple or brown spots on strawberry leaves.",
"Tomato___Bacterial_spot":"Bacterial spot causes small dark lesions on tomato leaves and fruit.",
"Tomato___Early_blight":"Early blight causes dark concentric-ring lesions on tomato leaves.",
"Tomato___Late_blight":"Late blight can produce dark, water-soaked lesions and spread quickly.",
"Tomato___Leaf_Mold":"Leaf mold commonly causes yellow patches and mold growth on the underside of leaves.",
"Tomato___Septoria_leaf_spot":"Septoria leaf spot causes many small circular spots on tomato leaves.",
"Tomato___Spider_mites_Two-spotted_spider_mite":"Two-spotted spider mites can cause yellow stippling and damage to leaves.",
"Tomato___Target_Spot":"Target spot causes circular lesions with concentric rings.",
"Tomato___Tomato_Yellow_Leaf_Curl_Virus":"This virus can cause curling, yellowing and stunted plant growth.",
"Tomato___Tomato_mosaic_virus":"This virus can cause mosaic patterns, leaf distortion and reduced plant growth."}

def clean_name(name): return name.replace("___"," – ").replace("_"," ")
def healthy(name): return "healthy" in name.lower()
def recommendation(name):
    return ("The plant appears healthy based on the model prediction. Continue regular watering, nutrition and routine monitoring." if healthy(name) else "Inspect the affected plant carefully. If appropriate, isolate it from healthy plants and consult an agricultural expert before applying treatment.")

# HERO
st.markdown('<div class="hero"><div class="badge">🌱 AI-POWERED AGRICULTURAL ASSISTANCE</div><h1>🌿 PlantGuard AI</h1><p>Detect possible plant diseases from a leaf image using an AI image classification model trained on the PlantVillage dataset.</p></div>',unsafe_allow_html=True)

# UPLOAD
st.markdown('<div class="title">📸 Upload a Leaf Image</div>',unsafe_allow_html=True)
st.markdown('<div class="sub">Upload a clear photo of a plant leaf and let PlantGuard AI analyze it.</div>',unsafe_allow_html=True)
st.markdown('<div class="upload"><h3>🌿 Choose your plant leaf image</h3><p>JPG, JPEG or PNG • Clear images generally provide better results</p>',unsafe_allow_html=True)
uploaded_file=st.file_uploader("Choose a leaf image",type=["jpg","jpeg","png"],label_visibility="collapsed")
st.markdown('</div>',unsafe_allow_html=True)

if uploaded_file is not None:
    image=Image.open(uploaded_file).convert("RGB")
    st.write("")
    preview,action=st.columns([1.05,.95],gap="large")
    with preview:
        st.markdown('<div class="card"><h3>📷 Your Leaf Image</h3><p>Image loaded and ready for analysis.</p></div>',unsafe_allow_html=True)
        st.write("")
        st.image(image,use_container_width=True)
        st.caption(f"📄 {uploaded_file.name}")
    with action:
        st.markdown('<div class="card"><h3>🔬 AI Analysis</h3><p>The trained image classification model will examine the uploaded leaf and estimate its most likely condition.</p></div>',unsafe_allow_html=True)
        st.write("")
        detect=st.button("🌿  Analyze Leaf",use_container_width=True)
        st.caption("🛡️ AI-assisted result • Human verification recommended")
        if detect:
            with st.spinner("🌱 PlantGuard AI is analyzing your leaf..."):
                # Keep model input processing unchanged.
                x=np.asarray(image.resize((160,160)),dtype=np.float32)
                x=tf.keras.applications.mobilenet_v2.preprocess_input(x)
                x=np.expand_dims(x,axis=0)
                predictions=model.predict(x,verbose=0)[0]
                idx=int(np.argmax(predictions)); conf=float(predictions[idx]); pred=CLASS_NAMES[idx]
                st.session_state["prediction"]=pred
                st.session_state["confidence"]=conf
            st.success("Analysis completed successfully.")

# RESULTS
if "prediction" in st.session_state:
    pred=st.session_state["prediction"]; conf=st.session_state["confidence"]; name=clean_name(pred)
    st.divider()
    st.markdown('<div class="title">🌱 Plant Health Result</div>',unsafe_allow_html=True)
    st.markdown('<div class="sub">AI-assisted analysis of your uploaded leaf.</div>',unsafe_allow_html=True)
    if healthy(pred):
        st.markdown(f'<div class="result-good"><div class="result-title">✅ Plant appears healthy</div><div class="small">{name}</div></div>',unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="result-review"><div class="result-title">⚠️ Possible condition detected</div><div class="small">{name}</div></div>',unsafe_allow_html=True)
    st.write("")
    c1,c2=st.columns([1,1],gap="large")
    with c1:
        st.markdown('<div class="card"><h3>📊 AI Confidence</h3></div>',unsafe_allow_html=True)
        st.metric("Model confidence",f"{conf*100:.1f}%")
        st.progress(min(max(conf,0),1))
    with c2:
        if conf<.60: st.warning("⚠️ **Low confidence**\n\nTry a clearer image and verify the result with an agricultural expert.")
        elif conf<.80: st.info("🔎 **Moderate confidence**\n\nUse this as an AI-assisted indication and consider human verification.")
        else: st.success("✅ **Higher confidence**\n\nThe model has relatively high confidence in its top prediction. Human verification is still recommended.")
    if pred in DISEASE_INFO:
        st.write("")
        st.markdown(f'<div class="card"><h3>💡 About this condition</h3><p>{DISEASE_INFO[pred]}</p></div>',unsafe_allow_html=True)
    st.write("")
    st.markdown(f'<div class="card"><h3>🧑‍🌾 Recommended next step</h3><p>{recommendation(pred)}</p></div>',unsafe_allow_html=True)
    st.write("")
    st.warning("🛡️ **Human verification matters**\n\nPlantGuard AI is an AI-assisted decision-support prototype, not a guaranteed diagnosis. Image quality, lighting, plant variety and conditions outside the training data can affect the result. Important treatment decisions should be reviewed by a farmer or agricultural expert.")

# HOW IT WORKS
st.write(""); st.divider()
st.markdown('<div class="title">⚙️ How It Works</div>',unsafe_allow_html=True)
st.markdown('<div class="sub">A simple three-step AI-assisted plant health workflow.</div>',unsafe_allow_html=True)
a,b,c=st.columns(3,gap="large")
with a: st.markdown('<div class="step"><h3>1️⃣ Upload</h3><p>Upload a clear image of a plant leaf.</p></div>',unsafe_allow_html=True)
with b: st.markdown('<div class="step"><h3>2️⃣ Analyze</h3><p>The trained AI image classification model analyzes the image.</p></div>',unsafe_allow_html=True)
with c: st.markdown('<div class="step"><h3>3️⃣ Review</h3><p>Review the prediction and verify important cases with an agricultural expert.</p></div>',unsafe_allow_html=True)

st.write("")
st.markdown('<div class="card"><h3>🛡️ Responsible AI</h3><p>PlantGuard AI provides an AI-assisted prediction rather than a guaranteed diagnosis. Results may be affected by image quality, lighting, plant variety and conditions not represented in the training data. Human review is recommended before important agricultural treatment decisions.</p></div>',unsafe_allow_html=True)
st.markdown('<div class="footer"><b>🌿 PlantGuard AI</b><br>AI-assisted plant health decision support • Prototype</div>',unsafe_allow_html=True)
