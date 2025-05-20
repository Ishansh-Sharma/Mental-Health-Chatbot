import streamlit as st
import sounddevice as sd
import numpy as np
import faster_whisper
import google.generativeai as genai

# Initialize the Whisper model
model = faster_whisper.WhisperModel("base")

genai.configure(api_key="AIzaSyA9w_lt-LcYIbTgq2rqZqe7KTZueHbnMbM")
model_llm = genai.GenerativeModel("gemini-1.5-flash")


# Function to process audio and transcribe speech to text
def transcribe_audio_to_text():
    print("Listening...")
    duration = 5  # Recording duration in seconds
    fs = 16000  # Sampling frequency
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="float32")
    sd.wait()  # Wait for the recording to finish
    audio_data = np.squeeze(recording)

    # Transcribing the audio
    segments, _ = model.transcribe(audio_data, beam_size=5)
    transcription = " ".join([segment.text for segment in segments])
    return transcription


# Function to send text to LLM and get a response
def get_response_from_llm(text):
    prompt = f'''
    Act as a mental health support chatbot. Provide empathetic, supportive responses that also feel like a reference to what they are typing to the following message:

    User message: "{text}"

    Give a thoughtful, compassionate response in a unique way that helps the user feel heard and supported.
    Offer practical coping strategies, but only when they ask for it.
    Encourage professional help only when they mention serious self-harm, like suicide. Never encourage it otherwise.
    Keep responses concise but helpful. Always ask in different and innovative ways to keep the conversation going.
    '''
    try:
        response = model_llm.generate_content(prompt)
        reply = response.text
        return reply
    except Exception as e:
        return "An error occurred while processing your request. Please try again."


# Streamlit UI setup
def main():
    st.set_page_config(page_title="Mental Health Support Chatbot", layout="wide")
    st.title("Mental Health Support Chatbot")

    # Sidebar for choosing the mode
    option = st.sidebar.radio("Choose mode:", ("Chatbot", "Voice Input"))

    if option == "Chatbot":
        st.header("Chat with the Mental Health Support Bot")
        chat_history = []

        with st.form(key="chat_form", clear_on_submit=True):
            user_input = st.text_input("You:", "")
            submit_button = st.form_submit_button(label="Send")

        if submit_button and user_input.strip():
            response = get_response_from_llm(user_input)
            chat_history.append(("You", user_input))
            chat_history.append(("Bot", response))

        # Display the chat history in a WhatsApp-like conversation UI
        for speaker, message in chat_history:
            if speaker == "You":
                st.markdown(
                    f'<div style="background-color: #DCF8C6; padding: 10px; margin: 5px; border-radius: 8px; width: fit-content;">{message}</div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div style="background-color: #FFFFFF; padding: 10px; margin: 5px; border-radius: 8px; width: fit-content;">{message}</div>',
                    unsafe_allow_html=True)

    elif option == "Voice Input":
        st.header("Voice Input for Command Processing")
        transcription = transcribe_audio_to_text()
        if transcription:
            st.write(f"Transcription: {transcription}")

            prompt = f'Text = "{transcription}". Based on the text provided, choose from the following directions: "up", "down", "left", "right", "forward", "backward". For example, "go upwards" would mean "upwards", "move forward" would mean "forward".'

            try:
                response = model_llm.generate_content(prompt)
                st.write(f"LLM Response: {response.text}")
            except Exception as e:
                st.write(f"Error: {e}")
        else:
            st.write("No speech detected. Try again.")


if __name__ == "__main__":
    main()
