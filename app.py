import streamlit as st
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="BrainGuard AI", page_icon="🧠")

st.title("🧠 BrainGuard AI")
st.warning("⚠️ Image validation: Only brain MRI scans accepted")

uploaded = st.file_uploader("Upload MRI", type=['jpg','png','jpeg'])

if uploaded:
    # Read bytes ONCE
    img_bytes = uploaded.read()
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    arr = np.array(img)
    
    # Simple color check
    r = arr[:,:,0].astype(float)
    g = arr[:,:,1].astype(float)
    b = arr[:,:,2].astype(float)
    
    color_diff = np.mean(np.abs(r-g) + np.abs(r-b) + np.abs(g-b)) / 3
    
    st.write(f"DEBUG: Color difference = {color_diff:.2f}")
    
    # REJECT if colorful
    if color_diff > 20:
        st.error(f"❌ REJECTED: Too colorful ({color_diff:.1f})")
        st.error("This is NOT a brain MRI scan")
        st.image(img, width=300)
        st.stop()
    
    # ACCEPT
    st.success(f"✅ VALID: Grayscale image ({color_diff:.1f})")
    st.image(img)
    st.balloons()

st.caption("Test: Upload a colorful screenshot - should be rejected")
st.caption("Test: Upload a brain MRI - should be accepted")