"""
Simple test for LiteLLM and OpenRouter providers
"""

import os
from chameleon.utils.pipeline_builder import ChameleonPipelineBuilder
from chameleon.utils.logging_utils import setup_colored_logger
from langchain_core.documents import Document

def test_openrouter():
    """Test OpenRouter provider."""
    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ OpenRouter API key not found")
        return False
    
    try:
        print("🧪 Testing OpenRouter...")
        
        # Sample documents as Document objects
        documents = [
            Document(page_content="LiteLLM provides a unified interface to over 100+ LLM providers."),
            Document(page_content="OpenRouter offers access to various AI models through a single API."),
            Document(page_content="The CHAMELEON framework supports multiple RAG pipeline types.")
        ]
        
        # Build pipeline
        pipeline = (ChameleonPipelineBuilder()
                   .with_openrouter("anthropic/claude-3.5-sonnet")
                   .with_faiss()
                   .with_basic_rag()
                   .with_documents(documents)
                   .build(validate=False))
        
        # Test query
        response = pipeline.run("What is LiteLLM?")
        print(f"✅ OpenRouter working! Response: {response['response'][:100]}...")
        return True
        
    except Exception as e:
        print(f"❌ OpenRouter failed: {str(e)}")
        return False

def test_litellm():
    """Test LiteLLM provider."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ Anthropic API key not found (needed for LiteLLM)")
        return False
    
    try:
        print("🧪 Testing LiteLLM...")
        
        # Sample documents as Document objects
        documents = [
            Document(page_content="LiteLLM provides a unified interface to over 100+ LLM providers."),
            Document(page_content="OpenRouter offers access to various AI models through a single API."),
            Document(page_content="The CHAMELEON framework supports multiple RAG pipeline types.")
        ]
        
        # Build pipeline
        pipeline = (ChameleonPipelineBuilder()
                   .with_litellm("anthropic/claude-3-sonnet-20240229")
                   .with_faiss()
                   .with_basic_rag()
                   .with_documents(documents)
                   .build(validate=False))
        
        # Test query
        response = pipeline.run("What is the CHAMELEON framework?")
        print(f"✅ LiteLLM working! Response: {response['response'][:100]}...")
        return True
        
    except Exception as e:
        print(f"❌ LiteLLM failed: {str(e)}")
        return False

def main():
    """Test the new providers."""
    setup_colored_logger()
    
    print("🚀 Testing New Provider Support")
    print("=" * 40)
    
    results = []
    
    # Test OpenRouter
    results.append(test_openrouter())
    
    # Test LiteLLM
    results.append(test_litellm())
    
    # Summary
    successful = sum(results)
    total = len(results)
    
    print(f"\n📊 Results: {successful}/{total} providers working")
    
    if successful > 0:
        print("🎉 New provider support is working!")
    else:
        print("⚠️ No providers could be tested (missing API keys)")

if __name__ == "__main__":
    main() 