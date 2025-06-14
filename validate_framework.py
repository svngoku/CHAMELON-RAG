#!/usr/bin/env python3
"""
CHAMELEON RAG Framework Validation Script
Comprehensive validation and setup verification for the framework.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Main validation function."""
    print("🔍 CHAMELEON RAG Framework Validation")
    print("=" * 50)
    
    try:
        # Import validation utilities
        from chameleon.utils.structure_validator import validate_chameleon_structure
        from chameleon.utils.pipeline_builder import ChameleonPipelineBuilder
        from chameleon.utils.health_checker import get_health_checker
        
        print("\n1️⃣ Running Structure Validation...")
        structure_report = validate_chameleon_structure()
        
        print("\n2️⃣ Testing Pipeline Builder...")
        builder = ChameleonPipelineBuilder()
        builder.list_templates()
        
        print("\n3️⃣ Testing Health Monitoring...")
        health_checker = get_health_checker()
        health_checker.register_component("validation_test", None)
        health_checker.check_all_components()
        health_summary = health_checker.get_health_summary()
        print(f"Health Status: {health_summary['overall_status']}")
        
        print("\n4️⃣ Checking API Keys...")
        api_keys = {
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "TOGETHER_API_KEY": os.getenv("TOGETHER_API_KEY")
        }
        
        for key, value in api_keys.items():
            status = "✅ Set" if value else "❌ Not set"
            print(f"  {key}: {status}")
        
        print("\n5️⃣ Framework Status Summary:")
        success_rate = structure_report['summary']['success_rate']
        
        if success_rate >= 95:
            print("🎉 Framework is EXCELLENT - Ready for production use!")
        elif success_rate >= 90:
            print("✅ Framework is GOOD - Ready for development use!")
        elif success_rate >= 80:
            print("⚠️ Framework is FUNCTIONAL - Minor issues to address")
        else:
            print("🚨 Framework needs ATTENTION - Multiple issues found")
        
        print(f"\nOverall Success Rate: {success_rate:.1f}%")
        
        if not any(api_keys.values()):
            print("\n💡 To test with real models, set your API keys:")
            print("   export OPENAI_API_KEY='your-key'")
            print("   export TOGETHER_API_KEY='your-key'")
        
        print("\n🚀 Quick Start:")
        print("   from chameleon.utils import quick_pipeline")
        print("   from langchain_core.documents import Document")
        print("   docs = [Document(page_content='test')]")
        print("   pipeline = quick_pipeline(docs)")
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Make sure all dependencies are installed:")
        print("   uv pip install -e .")
        sys.exit(1)
    
    except Exception as e:
        print(f"❌ Validation Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 