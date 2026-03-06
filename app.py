import warnings
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from utils import enhance_image, extract_text
import os

# suppress deprecation/future warnings from the google package
warnings.filterwarnings("ignore", category=FutureWarning)

# try to import the newer library if available, otherwise fall back
genai = None
try:
    import google.genai as genai
except ImportError:
    try:
        import google.generativeai as genai
    except ImportError:
        genai = None


# Configure Gemini API
if genai is not None:
    # prefer environment variable for key, fall back to hard-coded value
    api_key = os.getenv("GEMINI_API_KEY", "AIzaSyCdnzx8DmfY4XPQVJnHnJyyWViUM30Ol_Y")
    try:
        genai.configure(api_key=api_key)
    except Exception as exc:
        st.error(f"Failed to configure AI client: {exc}")
else:
    st.warning("Google AI SDK not installed; conversion will not work.")

st.title("📝 Scribble to Digital")
st.write("Convert messy handwritten notes into clean text & to-do lists")

uploaded_file = st.file_uploader("Upload notes image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(image)
    enhanced = enhance_image(img_array)

    st.image(enhanced, caption="Enhanced Image", use_column_width=True)

    with st.spinner("🔍 Extracting text with OCR..."):
        raw_text = extract_text(enhanced)

    st.subheader("📄 Raw OCR Text")
    st.text(raw_text)

    if st.button("✨ Convert to Digital"):
        with st.spinner("🤖 Processing with AI..."):
            prompt = f"""
            Clean this OCR text, correct spelling using context,
            and extract to-do tasks separately.

            OCR Text:
            {raw_text}

            Output format:
            Clean Notes:
            - ...

            To-Do List:
            - ...
            """

            if genai is not None:
                try:
                    # Use whichever model class is provided by the SDK
                    generator_cls = getattr(genai, 'GenerativeModel', None)
                    if generator_cls is None and hasattr(genai, 'Client'):
                        # some versions expose a client object
                        client = genai.Client()
                        generator_cls = client.generative_model
                    if generator_cls is None:
                        raise RuntimeError("AI model class not found in SDK")

                    generator = generator_cls('gemini-2.5-flash')
                    response = generator.generate_content(prompt)
                    result = getattr(response, 'text', str(response))
                    st.subheader("✅ Digital Output")
                    st.text(result.replace('\n', ' ').strip())
                except Exception as ai_exc:
                    st.error(f"AI processing failed: {ai_exc}")
            else:
                st.error("AI SDK not available, cannot convert text.")
