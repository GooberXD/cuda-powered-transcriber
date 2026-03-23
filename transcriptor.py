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


# Load the model once and cache it for future use
@st.cache_resource
def load_whisper_model():
    return WhisperModel(MODEL_SIZE, device=device, compute_type="float16")

# clear uploaded file and then reset app
def reset_app():
    for key in st.session_state.keys():
        del st.session_state[key]
    st.rerun()

st.title("Video Transcriber, powered by autism AI")
st.write("upload cum here and i spit that thang (text).")



# upload vid here
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0

uploaded_file = st.file_uploader("Choose a video file, neckhurt!!", type=["mp4", "mkv", "mov", "avi"],key=f"uploader_{st.session_state.uploader_key}")



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
    
    full_text = ""
    with st.spinner("Transcribing... (This may take a minute chat)"):
        model = load_whisper_model()
        segments, info = model.transcribe(audio_path, beam_size=5)
        
        # container to show text as it generates text
        text_container = st.container()
        for segment in segments:
            # prints and stores thine text with timestamps 
            # format eg. [11.92s -> 17.68s]
            timestamp = f"[{segment.start:.2f}s -> {segment.end:.2f}s]"
            line = f"{timestamp} {segment.text}\n"
            text_container.write(line)
            full_text += line

    # Downloads transcript to scripts director
    st.divider()
    original_filename = uploaded_file.name
    base_name = os.path.splitext(original_filename)[0]
    download_name = f"{base_name}_transcript.txt"
    
    
    st.download_button(
        label="Download Transcript",
        data=full_text,
        file_name=download_name,
        mime="text/plain"
    )
        
    if st.button("Transcribe New Video gng"):
        # Cleanup temp files before resetting
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
            if os.path.exists(audio_path):
                os.remove(audio_path)
        except:
            pass
        
        # inc key to reset uploader and rerun app
        st.session_state.uploader_key += 1
        st.rerun()

    # clean up automatic after transcription, incase user forgets to click the button
    try:
        if os.path.exists(video_path):
            os.remove(video_path)
        if os.path.exists(audio_path):
            os.remove(audio_path)
    except Exception as e:
        pass
        
        