import sounddevice as sd
import numpy as np
import faster_whisper
import google.generativeai as genai


model = faster_whisper.WhisperModel("base")

# Configure Gemini API
API_KEY = "API KEY"
genai.configure(api_key=API_KEY)
model_llm = genai.GenerativeModel("gemini-1.5-flash")


def transcribe_audio_to_text():
    print("I'm listening... Feel free to talk.")
    duration = 5  # Recording duration in seconds
    fs = 16000  # Sampling frequency
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="float32")
    sd.wait()
    audio_data = np.squeeze(recording)

    # Transcribe the audio
    segments, _ = model.transcribe(audio_data, beam_size=5)
    transcription = " ".join([segment.text for segment in segments])
    print(f"You said: {transcription}")
    return transcription


def get_response_from_llm(text):
    print("Thinking...")
    prompt = (f'You are a supportive and understanding mental health companion. Listen to the user and provide comforting, empathetic responses.If they seem distressed, reassure them.Keep the conversation flowing. User: "{text}" AI:')
    try:
        response = model_llm.generate_content(prompt)
        reply = response.text
        print(f"Your Friend: {reply}")
        return reply
    except Exception as e:
        print(f"Error while communicating with Gemini LLM: {e}")
        return "I'm here for you, but I encountered an issue understanding you. Can you say that again?"


if __name__ == "__main__":
    print("Hello! I'm here to listen. Talk to me whenever you need support.")
    while True:
        try:
            transcription = transcribe_audio_to_text()
            if transcription.strip():
                llm_response = get_response_from_llm(transcription)
            else:
                print("I couldn't catch that. Could you repeat?")
        except KeyboardInterrupt:
            print("\nGoodbye! Remember, you're not alone.")
            break
        except Exception as e:
            print(f"Error: {e}")
