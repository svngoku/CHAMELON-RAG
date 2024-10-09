from langchain_openai import OpenAIEmbeddings

class SimpleGenerator:
    def __init__(self, model_name: str = "text-davinci-003"):
        self.embeddings = OpenAIEmbeddings(model_name=model_name)

    def generate(self, context: str, query: str) -> str:
        # Placeholder for generation logic
        return f"Generated response for query: {query} with context: {context}"
