import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from the .env file
load_dotenv()

# Initialize the OpenAI client 
# (It automatically looks for the OPENAI_API_KEY environment variable)
client = OpenAI()

def main():
    print("Sending request to OpenAI...")
    
    try:
        # Make a simple chat completion request
        response = client.chat.completions.create(
            model="gpt-4o", 
            messages=[
                {"role": "system", "content": "You are a helpful and creative assistant."},
                {"role": "user", "content": "Write a one-sentence sci-fi version of 'Hello, World!'"}
            ]
        )
        
        # Output the response
        print("\nResponse:")
        print(response.choices[0].message.content)
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()