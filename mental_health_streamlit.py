import streamlit as st
import sounddevice as sd
import numpy as np
import faster_whisper
import google.generativeai as genai
import io


model = faster_whisper.WhisperModel("base")


API_KEY = "YOUR API KEY"
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

    try:
        
        segments, _ = model.transcribe(audio_data, beam_size=5)
        transcription = " ".join([segment.text for segment in segments])
        return transcription
    except Exception as e:
        st.error(f"Error during transcription: {e}")
        return "I couldn't catch that. Can you try again?"


def get_response_from_llm(text):
    if not text.strip():
        return "I didn't hear anything. Can you try again?"

    prompt = f"You are a supportive and understanding mental health companion. Listen to the user and provide appropriate responses according to the situation. If they seem distressed, reassure them a little .you can also offer solutions in short to their problems. Keep the conversation flowing and casually ask for follow ups so that the user feels engaged. if you have the word 'suicide' tell them to get professional help .  User: \"{text}\" AI:"
    try:
        response = model_llm.generate_content(prompt)
        reply = response.text
        return reply
    except Exception as e:
        st.error(f"Error with the LLM response: {e}")
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
            padding: 15px 40px;
            border-radius: 8px;
            width: 100%;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
        .stMarkdown {
            font-size: 18px;
        }
        .chat-container {
            max-width: 600px;
            margin: auto;
            padding: 20px;
            background-color: #06402B;
            border-radius: 15px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        }
        .user-message {
            background-color: #325716;
            padding: 10px;
            margin-bottom: 10px;
            border-radius: 8px;
            max-width: 80%;
            margin-left: auto;
        }
        .bot-message {
            background-color: #000000;
            padding: 10px;
            margin-bottom: 10px;
            border-radius: 8px;
            max-width: 80%;
            margin-right: auto;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar to choose between voice or chatbot
mode = st.sidebar.radio("Choose Mode", ["Voice Input", "Chatbot"])

# UI Layout
if mode == "Chatbot":
    st.header("Chat with me 👥")
    chat_history = []

    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("You:", "")
        submit_button = st.form_submit_button(label="Send")

    if submit_button and user_input.strip():
        response = get_response_from_llm(user_input)
        chat_history.append(("You", user_input))
        chat_history.append(("Bot", response))

    # Display the chat history in a WhatsApp-like conversation UI
    with st.container():
        for speaker, message in chat_history:
            if speaker == "You":
                st.markdown(f'<div class="user-message">{message}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="bot-message">{message}</div>', unsafe_allow_html=True)

elif mode == "Voice Input":
    st.header("Talk to Me 🎤")
    st.markdown("Click the button below to start speaking. I'll listen to you and provide responses.")

    record_button = st.button("Start Recording", help="Click to start recording your voice")

    if record_button:
        transcription = transcribe_audio_to_text()
        st.write(f"You said: **{transcription}**")

        if transcription.strip():  # If there's some transcription
            response = get_response_from_llm(transcription)
            st.write(f"Your Friend: **{response}**")
        else:
            st.write("I couldn't catch that. Could you try again?")

# Sidebar content
with st.sidebar:
    st.sidebar.header("How It Works")
    st.sidebar.markdown("""
        This is a simple mental health chatbot designed to listen to your voice and provide comforting responses. 
        Feel free to speak and share your feelings, and I will do my best to help and support you.
        If you need immediate help, please reach out to a professional.
    """)

# Footer
st.markdown("""
    <footer style="text-align:center; padding: 20px;">
        <p style="color: #888;">Developed with ❤️ by Team Random | <a href="mailto:support@example.com" style="color: #4CAF50;">Contact Us</a></p>
    </footer>
""", unsafe_allow_html=True)
