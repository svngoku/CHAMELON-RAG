from langchain_openai import OpenAIEmbeddings, OpenAILLM

class SimpleGenerator:
    def __init__(self, model_name: str = "text-davinci-003"):
        self.embeddings = OpenAIEmbeddings(model_name=model_name)

    def generate(self, context: str, query: str) -> str:
        llm = OpenAILLM(model_name=self.embeddings.model_name)
        response = llm.generate(context=context, query=query)
        return response
