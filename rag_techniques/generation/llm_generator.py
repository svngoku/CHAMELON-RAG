from rag_techniques.base import BaseGenerator
from langchain import LangchainClient
from langchain.models import GPT4o


class LLMGenerator(BaseGenerator):
    def __init__(self):
        self.client = LangchainClient(model=GPT4o())

    def process(self, data):
        # Process the data and generate a summary
        print("Processing data in LLMGenerator")
        summary = self.summarize(data)
        print("Summary:", summary)

    def summarize(self, data):
        """
        Summarize the retrieved data using Langchain with GPT-4o.
        :param data: The data to summarize.
        :return: A summary of the data.
        """
        response = self.client.summarize(data)
        return response['summary']
