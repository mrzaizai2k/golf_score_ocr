import sys
sys.path.append("")

import streamlit as st
import os
import json
from PIL import Image
# from golf_reader_old import GolfReader
from src.gemini_reader import GeminiReader
from src.qwen2_reader import Qwen2Reader

# Initialize GolfReader
config_path = "config/config.yaml"  # Adjust this path as needed
golf_reader = GeminiReader(config_path)

st.title("Golf Score Reader")

# File uploader for multiple images
uploaded_files = st.file_uploader("Choose image(s) of golf scorecards", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Create columns for image and JSON output
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(Image.open(uploaded_file), caption=uploaded_file.name, use_column_width=True)
        
        with col2:
            st.subheader(f"Results for {uploaded_file.name}")
            
            # Save the uploaded file temporarily
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            try:
                # Process the image and get results
                result = golf_reader.process_image(temp_path)
                
                # Display the JSON output
                st.json(result)
                 
            except Exception as e:
                st.error(f"An error occurred while processing {uploaded_file.name}: {str(e)}")
            
            finally:
                # Clean up the temporary file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        
        st.markdown("---")  # Add a separator between images

else:
    st.info("Please upload one or more images of golf scorecards to process.")

# Add some information about how to use the app
st.sidebar.header("How to Use")
st.sidebar.markdown("""
1. Click on 'Browse files' to select one or more images of golf scorecards from your local machine.
2. The app will process each image and display the results.
3. The original image will be shown on the left, and the extracted golf score data in JSON format will be displayed on the right.
4. If there are any errors in processing, they will be displayed below the respective image.
""")