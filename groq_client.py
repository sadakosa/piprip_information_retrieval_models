
import os
from groq import Groq
import json


class GroqClient:
    def __init__(self, api_key_input, logger):
        try:
            self.client = Groq(api_key=api_key_input)
            self.logger = logger
            print("Groq client initialized")
        except Exception as e:
            print(f"Error initializing Groq client: {e}")
            raise

    def query(self, user_input, temperature=0.34):
        try:
            chat_completion = self.client.chat.completions.create(
                messages=user_input,
                model="llama3-8b-8192",
                temperature=temperature,
                max_tokens=1024,
                stream=False,
                response_format={"type": "json_object"},
            )
            response = chat_completion.choices[0].message.content
            
            try:
                response_json = json.loads(response)
                return response_json
            except json.JSONDecodeError as json_err:
                print(f"JSON decode error: {json_err}")
                self.logger.log_message(f"JSON decode error: {json_err}")
                return {"error": "Invalid JSON response"}
        except Exception as e:
            print(f"Error during query: {e}")
            self.logger.log_message(f"Error during query: {e}")
            return {"error": "An error occurred during the query. Please try again later."}


    # def chatbot(self, system_instructions=""):
    #     conversation = [{"role": "system", "content": system_instructions}]
    #     while True:
    #         user_input = input("User: ")
    #         if user_input.lower() in ["exit", "quit"]:
    #             print("Exiting the chatbot. Goodbye!")
    #             break
    #         response, conversation = self.get_response(user_input, conversation)
    #         print(f"Assistant: {response}")







