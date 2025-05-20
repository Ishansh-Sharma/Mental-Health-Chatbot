import google.generativeai as genai

genai.configure(api_key="AIzaSyA9w_lt-LcYIbTgq2rqZqe7KTZueHbnMbM")
model_llm = genai.GenerativeModel("gemini-1.5-flash")

def get_response_from_llm(text):
    prompt = f'''
    Act as a mental health support chatbot. Provide empathetic, supportive responses that also feel like a referenc to what they are typing to the following message:
    
    User message: "{text}"
    
    Give a thoughtful, compassionate response in a unique way that helps the user feel heard and supported.
    Offer practical coping strategies , but only when they ask for it  .
     encourage professional help only and only when they mention about serious self harm lke suicide , never another time . 
    Keep responses concise but helpful.always ask in different and innovative way to keep the conversation going . 
    '''

    try:
        response = model_llm.generate_content(prompt)
        reply = response.text
        return reply
    except Exception as e:
        return "An error occurred while processing your request. Please try again."

def main():
    print("Mental Health Support Chatbot")
    print("-" * 50)

    while True:
        user_input = input("\nYou: ")
        response = get_response_from_llm(user_input)
        print("\nBot:", response)
        print("-" * 50)

if __name__ == "__main__":
    main()
