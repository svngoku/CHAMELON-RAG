from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_cohere import ChatCohere
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from typing import Dict, Any


class SimpleGenerator:
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.7):
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        self.prompt = ChatPromptTemplate.from_template(
            """Based on the following context, please answer the query.
            
            Context: {context}
            
            Query: {question}
            
            Answer:"""
        )
        self.chain = self.prompt | self.llm | StrOutputParser()

    def invoke(self, input_data: Dict[str, Any]) -> str:
        """Process the input data and generate a response."""
        try:
            return self.chain.invoke(input_data)
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def process(self, context: str, query: str) -> str:
        """Alternative interface for processing context and query."""
        return self.invoke({"context": context, "question": query})