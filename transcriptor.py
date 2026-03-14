import streamlit as st
import moviepy as mp
import os
import sys
import shutil

# dll loading
if sys.platform == 'win32':
    import site
    # Get all possible site-packages paths
    site_packages = site.getsitepackages()
    if site.getusersitepackages():
        site_packages.append(site.getusersitepackages())
    
    for base_path in site_packages:
        nvidia_base = os.path.join(base_path, 'nvidia')
        if os.path.exists(nvidia_base):
            # Look for cublas and cudnn bin folders
            for root, dirs, files in os.walk(nvidia_base):
                if 'bin' in dirs:
                    bin_path = os.path.join(root, 'bin')
                    # Check if the required dll is actually in this bin folder
                    if any("cublas64_12.dll" in f for f in os.listdir(bin_path)):
                        os.add_dll_directory(bin_path)
                        print(f"Added DLL directory: {bin_path}")
                        
# highkey if this doesnt work , copy paste the following files into the script's folder
# basically do this if CUDA errors persist

# cublas64_12.dll
# cublasLt64_12.dll
# cudnn64_9.dll
# zlipwapi.dll //personally didnt use this one but it still worked, so maybe its optional?


from faster_whisper import WhisperModel

# config
#* Options: tiny, base, small, medium, large-v3, 
#** Change to "cuda" if you have an NVIDIA GPU, "cpu" if none

MODEL_SIZE = "large-v3"  #*
device = "cuda" #** 

st.title("Video Transcriber, powered by autism AI")
st.write("upload cum here and i spit that thang (text).")

# upload vid here
uploaded_file = st.file_uploader("Choose a video file, neckhurt!!", type=["mp4", "mkv", "mov", "avi"])

if uploaded_file is not None:
    # Saves temp video and file for processing
    video_path = "temp_video.mp4"
    audio_path = "temp_audio.mp3"
    
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # audio extraction
    with st.status("Extracting audio...", expanded=True) as status:
        # Using 'with' automatically closes the file when the block ends
        with mp.VideoFileClip(video_path) as video:
            video.audio.write_audiofile(audio_path)
        status.update(label="Audio extracted!", state="complete")

    # transcribing section
    st.subheader("Transcription")
    
    with st.spinner("Transcribing... (This may take a minute chat)"):
        model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")
        segments, info = model.transcribe(audio_path, beam_size=5)
        
        full_text = ""
        for segment in segments:
            # prints and stores thine text with timestamps 
            # format eg. [11.92s -> 17.68s]
            timestamp = f"[{segment.start:.2f}s -> {segment.end:.2f}s]"
            line = f"{timestamp} {segment.text}\n"
            st.write(line)
            full_text += line

    # Downloads transcript to scripts director
    original_filename = uploaded_file.name
    base_name = os.path.splitext(original_filename)[0]
    
    download_name = f"{base_name}_transcript.txt"
    
    
    st.download_button(
        label="Download Transcript",
        data=full_text,
        file_name=download_name,
        mime="text/plain"
    )

    # Cleanup temporary files
    try:
        if os.path.exists(video_path):
            os.remove(video_path)
        if os.path.exists(audio_path):
            os.remove(audio_path)
    except Exception as e:
        st.warning(f"Note: Could not delete temporary files, delete them manually pls: {e}")