
import sys
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)

# Mock openai for testing without keys
from unittest.mock import MagicMock
import sys
sys.modules["openai"] = MagicMock()
sys.modules["openai"].OpenAI = MagicMock()

# Now import wrapper involving agentdojo which uses openai
try:
    import benchmarks.external.agentdojo_wrapper as wrapper
    from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, PipelineConfig
    print("✅ Successfully imported agentdojo_wrapper")
except ImportError as e:
    print(f"❌ Failed to import wrapper: {e}")
    sys.exit(1)

def test_pipeline_creation():
    print("Testing pipeline creation with 'bit_guard'...")
    from agentdojo.models import ModelsEnum
    # Mock MODEL_PROVIDERS to allow our dummy model logic without hitting API
    from agentdojo.models import MODEL_PROVIDERS
    # Force a provider that we can control or just let it hit the mocked openai
    
    try:
        # Create a dummy config
        valid_model = list(ModelsEnum)[0].value
        print(f"Using model: {valid_model}")
        config = PipelineConfig(
            llm=valid_model, # Use valid model
            model_id=None,
            defense="bit_guard",
            system_message_name="default",
            system_message=None
        )
        
        # This calls the patched from_config
        # Note: This might try to initialize the real InjectionDetector which loads models.
        # Ensure we can handle that or mock it if needed. 
        # For this test, we expect it to try and fail or succeed if models are present.
        pipeline = AgentPipeline.from_config(config)
        
        print(f"✅ Pipeline created successfully: {pipeline.name}")
        
        # Verify structure
        has_detector = False
        for element in pipeline.elements:
            # The structure is SystemMessage -> InitQuery -> LLM -> ToolsLoop
            # Inside ToolsLoop -> [Executor, Detector, LLM]
            # We need to dig into ToolsLoop
            if hasattr(element, "elements"): # ToolsExecutionLoop
                for inner_el in element.elements:
                     if isinstance(inner_el, wrapper.BitGuardDetector):
                         has_detector = True
                         print("✅ Found BitGuardDetector in pipeline")
        
        if not has_detector:
            print("❌ BitGuardDetector NOT found in pipeline elements")
            # print elements for debugging
            for element in pipeline.elements:
                print(f" - {element}")
                if hasattr(element, "elements"):
                     for inner_el in element.elements:
                         print(f"   - {inner_el}")

    except Exception as e:
        print(f"❌ Pipeline creation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pipeline_creation()
