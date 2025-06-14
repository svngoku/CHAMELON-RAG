from typing import List, Dict, Any, Optional, Union, Callable
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool, Tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from ..base import BasePreprocessor


class ToolExecutor(BasePreprocessor):
    """
    Tool integration for RAG pipelines based on LangChain tools.
    
    Features:
    - Support for external API calls and functions
    - Dynamic tool selection based on query content
    - Result integration with retrieved documents
    
    Based on LangChain tools and agents:
    https://python.langchain.com/docs/modules/agents/tool_use/
    """
    
    def __init__(
        self,
        tools: List[BaseTool],
        llm: Optional[BaseLanguageModel] = None,
        auto_invoke: bool = True
    ):
        """
        Initialize the tool executor.
        
        Args:
            tools: List of LangChain tools to make available
            llm: Language model for tool selection and invocation
            auto_invoke: Whether to automatically invoke tools based on query
        """
        self.tools = tools
        self.llm = llm or ChatOpenAI(temperature=0)
        self.auto_invoke = auto_invoke
        
        # Create tool selection prompt
        self.tool_selector_prompt = ChatPromptTemplate.from_template(
            """You are an AI assistant that decides whether a query requires using a tool.
            
            Available tools:
            {tool_descriptions}
            
            Based on the user query, determine which tool (if any) should be used.
            If a tool should be used, respond with ONLY the name of the tool.
            If no tool is needed, respond with "NONE".
            
            User query: {query}
            
            Tool to use:"""
        )
        
        # Create tool invocation prompt
        self.tool_invoker_prompt = ChatPromptTemplate.from_template(
            """You are an AI assistant that helps users by calling tools.
            
            Tool: {tool_name}
            Tool description: {tool_description}
            
            Based on the user query, provide the input that should be passed to this tool.
            Respond with ONLY the exact input to pass to the tool, nothing else.
            
            User query: {query}
            
            Tool input:"""
        )
    
    def process(self, query: str) -> Dict[str, Any]:
        """
        Process the query and optionally invoke relevant tools.
        
        Args:
            query: The user's query
            
        Returns:
            Dictionary with processed results
        """
        result = {
            "query": query,
            "tool_used": None,
            "tool_result": None
        }
        
        # Skip tool invocation if auto_invoke is disabled
        if not self.auto_invoke:
            return result
        
        # Get tool descriptions
        tool_descriptions = "\n".join([
            f"- {tool.name}: {tool.description}" 
            for tool in self.tools
        ])
        
        # Determine which tool to use
        tool_selector_chain = (
            self.tool_selector_prompt 
            | self.llm 
            | StrOutputParser()
        )
        
        tool_name = tool_selector_chain.invoke({
            "tool_descriptions": tool_descriptions,
            "query": query
        }).strip()
        
        # If no tool needed, return
        if tool_name.upper() == "NONE":
            return result
        
        # Find the tool
        selected_tool = None
        for tool in self.tools:
            if tool.name.lower() == tool_name.lower():
                selected_tool = tool
                break
        
        # If tool not found, return
        if not selected_tool:
            return result
        
        # Generate tool input
        tool_invoker_chain = (
            self.tool_invoker_prompt 
            | self.llm 
            | StrOutputParser()
        )
        
        tool_input = tool_invoker_chain.invoke({
            "tool_name": selected_tool.name,
            "tool_description": selected_tool.description,
            "query": query
        }).strip()
        
        # Execute the tool
        try:
            tool_result = selected_tool.invoke(tool_input)
            
            # Update result
            result["tool_used"] = selected_tool.name
            result["tool_result"] = tool_result
            
        except Exception as e:
            result["tool_error"] = str(e)
        
        return result
    
    @classmethod
    def from_functions(
        cls,
        functions: List[Callable],
        llm: Optional[BaseLanguageModel] = None,
        auto_invoke: bool = True
    ) -> "ToolExecutor":
        """
        Create a ToolExecutor from a list of Python functions.
        
        Args:
            functions: List of functions to convert to tools
            llm: Language model for tool selection
            auto_invoke: Whether to automatically invoke tools
            
        Returns:
            Configured ToolExecutor
        """
        tools = []
        
        for func in functions:
            # Get function metadata
            name = func.__name__
            description = func.__doc__ or f"Function {name}"
            
            # Create tool
            tool = Tool(
                name=name,
                description=description,
                func=func
            )
            
            tools.append(tool)
        
        return cls(tools=tools, llm=llm, auto_invoke=auto_invoke)