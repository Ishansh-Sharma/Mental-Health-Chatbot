import pandas as pd

# Load the dataset
df = pd.read_csv('MDD-5K.csv')  # or .json, adjust accordingly

# Combine question/response or conversation entries if needed
df['text'] = df['dialogue']  # replace with actual column if different
corpus = df['text'].tolist()

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
