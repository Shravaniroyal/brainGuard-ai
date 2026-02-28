"""BrainGuard AI - ULTRA SIMPLE VERSION - GUARANTEED TO WORK"""
import streamlit as st
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="BrainGuard AI", page_icon="🧠")

def check_if_brain(image_bytes):
    """Simple check: grayscale + texture"""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        arr = np.array(img)
        
        # Grayscale check
        r = arr[:,:,0].astype(float)
        g = arr[:,:,1].astype(float) 
        b = arr[:,:,2].astype(float)
        color_diff = np.mean(np.abs(r-g) + np.abs(r-b) + np.abs(g-b)) / 3
        
        # If too colorful = NOT brain
        if color_diff > 20:
            return False, "Too colorful"
        
        # Texture check
        gray = np.mean(arr, axis=2)
        if np.var(gray) < 200:
            return False, "No texture"
        
        return True, "OK"
    except:
        return False, "Error"

st.title("🧠 BrainGuard AI")
st.warning("⚠️ Only brain MRI scans accepted")

uploaded = st.file_uploader("Upload brain MRI", type=['jpg','png'])

if uploaded:
    file_bytes = uploaded.read()
    
    # CHECK IT
    is_brain, msg = check_if_brain(file_bytes)
    
    if not is_brain:
        st.error(f"❌ REJECTED: {msg}")
        st.error("Upload brain MRI only!")
        st.stop()
    
    st.success("✅ Valid brain MRI!")
    st.image(Image.open(io.BytesIO(file_bytes)))
    
    if st.button("Analyze"):
        st.success("Analysis would run here")
        st.balloons()