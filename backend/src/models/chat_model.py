import openai
from src.core.config import config

class ChatModel:
    def __init__(self):
        # Initialize OpenAI API key
        openai.api_key = config.OPENAI_API_KEY
        self.model = "gpt-4"
    
    def generate_response(self, context: str, question: str) -> str:
        """
        Generate a response using GPT-4 based on the provided context and question.

        Args:
        - context (str): The context extracted from the contract.
        - question (str): The user's legal question.

        Returns:
        - str: The answer generated by the model.
        """
        # Construct the prompt to guide the LLM
        prompt = (
            "You are a highly knowledgeable legal advisor. "
            "Given the following context from a legal contract, answer the user's question as precisely as possible.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

        try:
            # Call OpenAI's GPT-4 model
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3,  # Lower temperature for more deterministic responses
                n=1
            )
            
            # Extract the generated answer from the response
            return response.choices[0].message['content'].strip()
        
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I'm sorry, I couldn't process your request at the moment."

# Example Usage
if __name__ == "__main__":
    chat_model = ChatModel()
    context_example = "This agreement is made between Company A and Company B, detailing the terms of service..."
    question_example = "What are the termination conditions in this contract?"
    answer = chat_model.generate_response(context_example, question_example)
    print("Response:", answer)
