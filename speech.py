import sounddevice as sd
import numpy as np
import faster_whisper
import google.generativeai as genai

# Initialize the Whisper model
model = faster_whisper.WhisperModel("base")

genai.configure(api_key="API KEY")
model_llm = genai.GenerativeModel("gemini-1.5-flash")

# Function to process audio and transcribe speech to text
def transcribe_audio_to_text():
    print("i am listening to you ")
    duration = 5  # Recording duration in seconds
    fs = 16000  # Sampling frequency
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="float32")
    sd.wait()  # Wait for the recording to finish
    audio_data = np.squeeze(recording)

    # Transcribing the audio
    segments, _ = model.transcribe(audio_data, beam_size=5)
    transcription = " ".join([segment.text for segment in segments])
    print(f"Transcription: {transcription}")
    return transcription


# Function to send text to LLM and get a response
def get_response_from_llm(text):
    print("Sending text to LLM...")
    prompt=f'Text = \"{text}\".Based on the text provided choose from the following choices \"up\", \"down\", \"left\", \"right\", \"forward\", \"backward\". For example \"go upwards\" would mean \"upwards\", \"move forward\" would mean \"forward\"'
    try:
        response = model_llm.generate_content(
            prompt
        )
        reply = response.text
        print(f"LLM Response: {reply}")
        return reply
    except Exception as e:
        print(f"Error while communicating with Gemini LLM: {e}")
        return "An error occurred while processing your request."


# Main Function
if __name__ == "__main__":
    while True:
        try:
            transcription = transcribe_audio_to_text()
            if transcription.strip():  # Check if transcription is not empty
                llm_response = get_response_from_llm(transcription)
                print(f"AI Reply: {llm_response}")
            else:
                print("No speech detected. Try again.")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
