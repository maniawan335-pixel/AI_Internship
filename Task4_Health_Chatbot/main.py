from google import genai

# Gemini API key (OK for task demo; remove before GitHub)
client = genai.Client(api_key=".")

SYSTEM_PROMPT = """
You are a helpful and friendly medical assistant.
You provide general health-related information only.
You do not diagnose diseases or prescribe medication.
If a question involves serious symptoms, advise consulting a healthcare professional.
Give only to the point answer, dont give lenghty answers until it is asked
"""

def ask_health_bot(question):
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                SYSTEM_PROMPT,
                f"User question: {question}"
            ]
        )
        return response.text

    except Exception as e:
       return f"Error from Gemini API: {str(e)}"

if __name__ == "__main__":
    print("Welcome to the Health Assistant Bot (Gemini)\n")

    while True:
        q = input("Ask a health question (or type 'exit'): ")
        if q.lower() == "exit":
            print("Goodbye!")
            break
        print("\nBot:", ask_health_bot(q), "\n")