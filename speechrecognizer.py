import streamlit as st
import sounddevice as sd
import numpy as np
import speech_recognition as sr
from scipy.io import wavfile
import tempfile
import os
import base64

# Function to recognize speech from audio data
def recognize_speech(audio_data):
    recognizer = sr.Recognizer()
    try:
        text = recognizer.recognize_google(audio_data, language="en-US")
        return text
    except sr.UnknownValueError:
        return "Google Speech Recognition could not understand audio"
    except sr.RequestError as e:
        return f"Could not request results from Google Speech Recognition service; {e}"

# Function to create a download link for a text file
def create_download_link(text, filename="recognized_text.txt"):
    b64 = base64.b64encode(text.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">Download recognized text</a>'
    return href

# Streamlit UI
st.title("Real-Time Voice to Text Recognition")

# Parameters for audio recording
sample_rate = 16000  # Sample rate of the microphone

# Initialize reactive variables
if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = []
if 'recognized_text' not in st.session_state:
    st.session_state.recognized_text = ""

# UI components
start_button = st.button("Start Recording", key="start_button")

if start_button and not st.session_state.is_recording:
    st.session_state.is_recording = True
    st.session_state.audio_data = []  # Clear previous recording data

# Placeholder to show recognized speech
recognized_text_placeholder = st.empty()

# Record audio function
def record_audio(duration_seconds, sample_rate):
    audio_data = sd.rec(int(duration_seconds * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    return audio_data.flatten()

# Check if stop button is clicked
if st.session_state.is_recording:
    stop_button = st.button("Stop Recording", key="stop_button")
    if stop_button:
        st.session_state.is_recording = False
        
        if st.session_state.audio_data:
            # Concatenate all audio data chunks into a single numpy array
            audio_data_np = np.concatenate(st.session_state.audio_data)

            # Save audio_data to a temporary WAV file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_wav_file:
                temp_wav_filename = temp_wav_file.name
                wavfile.write(temp_wav_filename, sample_rate, audio_data_np.astype(np.int16))

            # Display the WAV file using st.audio
            st.audio(temp_wav_filename, format='audio/wav')

            # Convert audio_data to AudioData for speech recognition
            sample_width = audio_data_np.dtype.itemsize
            audio_data_for_recognition = sr.AudioData(audio_data_np.tobytes(), sample_rate=sample_rate, sample_width=sample_width)

            # Perform speech recognition on the entire recorded audio
            st.session_state.recognized_text = recognize_speech(audio_data_for_recognition, sample_rate)

            # Display recognized text
            recognized_text_placeholder.write(f"Recognized Text: {st.session_state.recognized_text}")

            # Delete the temporary WAV file after displaying
            os.remove(temp_wav_filename)
        else:
            recognized_text_placeholder.write("No audio data recorded.")

# If recording is ongoing, continue recording in small chunks
if st.session_state.is_recording:
    audio_chunk = record_audio(1, sample_rate)  # Record for a short duration
    st.session_state.audio_data.append(audio_chunk)
    st.experimental_rerun()

# Provide a download link for the recognized text
if st.session_state.recognized_text:
    download_link = create_download_link(st.session_state.recognized_text)
    st.markdown(download_link, unsafe_allow_html=True)
