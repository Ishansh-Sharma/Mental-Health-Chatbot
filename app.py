import streamlit as st
import sounddevice as sd
import numpy as np
import faster_whisper
import google.generativeai as genai
import io

# Initialize Whisper model for transcription
model = faster_whisper.WhisperModel("base")

# Configure Gemini API for LLM
API_KEY = "YOUR_API_KEY"
genai.configure(api_key=API_KEY)
model_llm = genai.GenerativeModel("gemini-1.5-flash")

# Function to transcribe audio
def transcribe_audio_to_text():
    st.write("I'm listening... Feel free to talk.")
    duration = 5  # Recording duration in seconds
    fs = 16000  # Sampling frequency
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="float32")
    sd.wait()
    audio_data = np.squeeze(recording)

    # Transcribe the audio using Whisper model
    segments, _ = model.transcribe(audio_data, beam_size=5)
    transcription = " ".join([segment.text for segment in segments])
    return transcription

# Function to get response from LLM (Gemini)
def get_response_from_llm(text):
    prompt = f"You are a supportive and understanding mental health companion. Listen to the user and provide comforting, empathetic responses. If they seem distressed, reassure them. Keep the conversation flowing. User: \"{text}\" AI:"
    try:
        response = model_llm.generate_content(prompt)
        reply = response.text
        return reply
    except Exception as e:
        return "I'm here for you, but I encountered an issue understanding you. Can you say that again?"

# Streamlit UI
st.set_page_config(page_title="Mental Health Support Chatbot", page_icon="💬", layout="wide")

# Title and Introduction
st.title("🧠 Mental Health Support Chatbot")
st.markdown("I'm here to listen and provide support. Feel free to share what's on your mind, and I'll respond with care and empathy.")

# Styling for aesthetic appearance
st.markdown("""
    <style>
        .stTextInput input {
            background-color: #f4f4f9;
            color: #333;
            font-size: 16px;
            padding: 10px;
            border-radius: 8px;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            font-size: 18px;
            padding: 10px 20px;
            border-radius: 8px;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
        .stMarkdown {
            font-size: 18px;
        }
    </style>
""", unsafe_allow_html=True)

# UI Layout
col1, col2 = st.columns([3, 1])

with col1:
    # Transcription button and display area
    st.header("Talk to Me 🎤")
    record_button = st.button("Start Recording")

    if record_button:
        transcription = transcribe_audio_to_text()
        st.write(f"You said: **{transcription}**")

        if transcription.strip():  # If there's some transcription
            response = get_response_from_llm(transcription)
            st.write(f"Your Friend: **{response}**")
        else:
            st.write("I couldn't catch that. Could you try again?")

with col2:
    # Sidebar for additional information and options
    st.sidebar.header("How It Works")
    st.sidebar.markdown("""
        This is a simple mental health chatbot designed to listen to your voice and provide comforting responses. 
        Feel free to speak and share your feelings, and I will do my best to help and support you.
        If you need immediate help, please reach out to a professional.
    """)

# Footer
st.markdown("""
    <footer style="text-align:center; padding: 20px;">
        <p style="color: #888;">Developed with ❤️ by Your Support Team " Random " | <a href="mailto:support@example.com" style="color: #4CAF50;">Contact Us</a></p>
    </footer>
""", unsafe_allow_html=True)
