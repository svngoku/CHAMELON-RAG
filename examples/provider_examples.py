"""
CHAMELEON RAG Framework - Provider Examples

This example demonstrates how to use different LLM providers with the CHAMELEON framework,
including the new LiteLLM and OpenRouter support.
"""

import os
from chameleon.utils.pipeline_builder import ChameleonPipelineBuilder, Provider, VectorStore, RAGType
from chameleon.utils.logging_utils import setup_colored_logger

def setup_environment():
    """Setup environment variables for different providers."""
    print("🔧 Setting up environment variables...")
    
    # Required API keys for different providers
    required_keys = {
        "OpenAI": "OPENAI_API_KEY",
        "Together AI": "TOGETHER_API_KEY", 
        "Groq": "GROQ_API_KEY",
        "Mistral": "MISTRAL_API_KEY",
        "Cohere": "COHERE_API_KEY",
        "Google": "GOOGLE_API_KEY",
        "OpenRouter": "OPENROUTER_API_KEY"
    }
    
    available_providers = []
    for provider, key in required_keys.items():
        if os.getenv(key):
            available_providers.append(provider)
            print(f"✅ {provider} API key found")
        else:
            print(f"❌ {provider} API key not found ({key})")
    
    return available_providers

def example_openai():
    """Example using OpenAI GPT models."""
    print("\n🤖 OpenAI Example")
    
    pipeline = (ChameleonPipelineBuilder()
                .with_openai("gpt-4o-mini")
                .with_faiss()
                .with_basic_rag()
                .build())
    
    return pipeline

def example_together_ai():
    """Example using Together AI models."""
    print("\n🤖 Together AI Example")
    
    pipeline = (ChameleonPipelineBuilder()
                .with_together("meta-llama/Llama-3.3-70B-Instruct-Turbo")
                .with_chroma()
                .with_contextual_rag()
                .build())
    
    return pipeline

def example_groq():
    """Example using Groq models."""
    print("\n🤖 Groq Example")
    
    pipeline = (ChameleonPipelineBuilder()
                .with_groq("llama3-8b-8192")
                .with_faiss()
                .with_multi_query_rag()
                .build())
    
    return pipeline

def example_litellm_anthropic():
    """Example using LiteLLM with Anthropic Claude."""
    print("\n🤖 LiteLLM + Anthropic Example")
    
    # Set Anthropic API key for LiteLLM
    os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY", "")
    
    pipeline = (ChameleonPipelineBuilder()
                .with_litellm("anthropic/claude-3-sonnet-20240229")
                .with_chroma()
                .with_parent_document_rag()
                .build())
    
    return pipeline

def example_litellm_gemini():
    """Example using LiteLLM with Google Gemini."""
    print("\n🤖 LiteLLM + Gemini Example")
    
    pipeline = (ChameleonPipelineBuilder()
                .with_litellm("gemini/gemini-pro")
                .with_faiss()
                .with_basic_rag()
                .build())
    
    return pipeline

def example_openrouter():
    """Example using OpenRouter."""
    print("\n🤖 OpenRouter Example")
    
    pipeline = (ChameleonPipelineBuilder()
                .with_openrouter("anthropic/claude-3.5-sonnet")
                .with_chroma()
                .with_contextual_rag()
                .build())
    
    return pipeline

def example_openrouter_llama():
    """Example using OpenRouter with Llama model."""
    print("\n🤖 OpenRouter + Llama Example")
    
    pipeline = (ChameleonPipelineBuilder()
                .with_openrouter("meta-llama/llama-3.1-405b-instruct")
                .with_faiss()
                .with_multi_query_rag()
                .build())
    
    return pipeline

def test_pipeline(pipeline, name: str):
    """Test a pipeline with sample data."""
    print(f"\n🧪 Testing {name} pipeline...")
    
    try:
        # Sample documents
        documents = [
            "The CHAMELEON framework is a flexible RAG system that supports multiple providers.",
            "LiteLLM provides a unified interface to over 100+ LLM providers.",
            "OpenRouter offers access to various AI models through a single API.",
            "RAG (Retrieval-Augmented Generation) combines retrieval and generation for better responses."
        ]
        
        # Add documents to pipeline
        pipeline.add_documents(documents)
        
        # Test query
        query = "What is the CHAMELEON framework?"
        response = pipeline.run(query)
        
        print(f"✅ {name} pipeline working!")
        print(f"Response: {response['response'][:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ {name} pipeline failed: {str(e)}")
        return False

def main():
    """Main function to demonstrate all providers."""
    setup_colored_logger()
    
    print("🚀 CHAMELEON RAG Framework - Provider Examples")
    print("=" * 60)
    
    # Setup environment
    available_providers = setup_environment()
    
    # Provider examples
    examples = {
        "OpenAI": (example_openai, "OPENAI_API_KEY"),
        "Together AI": (example_together_ai, "TOGETHER_API_KEY"),
        "Groq": (example_groq, "GROQ_API_KEY"),
        "LiteLLM + Anthropic": (example_litellm_anthropic, "ANTHROPIC_API_KEY"),
        "LiteLLM + Gemini": (example_litellm_gemini, "GOOGLE_API_KEY"),
        "OpenRouter": (example_openrouter, "OPENROUTER_API_KEY"),
        "OpenRouter + Llama": (example_openrouter_llama, "OPENROUTER_API_KEY")
    }
    
    successful_tests = 0
    total_tests = 0
    
    for name, (example_func, required_key) in examples.items():
        if os.getenv(required_key):
            try:
                pipeline = example_func()
                if test_pipeline(pipeline, name):
                    successful_tests += 1
                total_tests += 1
            except Exception as e:
                print(f"❌ Failed to create {name} pipeline: {str(e)}")
                total_tests += 1
        else:
            print(f"⏭️  Skipping {name} (no API key)")
    
    print(f"\n📊 Results: {successful_tests}/{total_tests} providers working")
    
    # Provider comparison
    print("\n📋 Provider Comparison:")
    print("┌─────────────────────┬─────────────────┬─────────────────────┐")
    print("│ Provider            │ Best For        │ Key Features        │")
    print("├─────────────────────┼─────────────────┼─────────────────────┤")
    print("│ OpenAI              │ General use     │ Reliable, fast      │")
    print("│ Together AI         │ Open models     │ Cost-effective      │")
    print("│ Groq                │ Speed           │ Ultra-fast inference│")
    print("│ LiteLLM             │ Flexibility     │ 100+ providers      │")
    print("│ OpenRouter          │ Model variety   │ Many models, one API│")
    print("└─────────────────────┴─────────────────┴─────────────────────┘")

if __name__ == "__main__":
    main() 