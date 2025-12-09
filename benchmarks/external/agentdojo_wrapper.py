
import sys
import os
from pathlib import Path
import logging
import json
from functools import partial

# Add AgentDojo to path
current_dir = Path(__file__).parent.absolute()
agentdojo_src = current_dir / "agentdojo" / "src"
sys.path.append(str(agentdojo_src))

# Add project root to path
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from agentdojo.agent_pipeline.pi_detector import PromptInjectionDetector
# Import AgentPipeline components and constants
from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, get_llm, DEFENSES, TOOL_FILTER_PROMPT
from agentdojo.agent_pipeline.basic_elements import InitQuery, SystemMessage
from agentdojo.agent_pipeline.tool_execution import ToolsExecutionLoop, ToolsExecutor, tool_result_to_str
from agentdojo.models import MODEL_PROVIDERS, ModelsEnum
from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLM, OpenAILLMToolFilter

# Import our detector
try:
    from src.detection.ensemble import InjectionDetector
except ImportError as e:
    logging.error(f"Failed to import InjectionDetector: {e}")
    # Fallback or exit
    raise

class BitGuardDetector(PromptInjectionDetector):
    """
    Wrapper for BIT InjectionDetector to be used in AgentDojo.
    """
    def __init__(self, mode="message", raise_on_injection=False, threshold=0.5):
        super().__init__(mode=mode, raise_on_injection=raise_on_injection)
        logging.info("Initializing BIT InjectionDetector...")
        # Initialize our detector
        # We assume models are already trained or will rely on default/mock if not found
        # In real eval, they should be trained.
        self.detector = InjectionDetector(
            model_dir=str(project_root / "models"),
            fast_threshold=threshold, # Using threshold from config if passed
            deep_threshold=0.95 
        )
        self.detector.is_trained = True # Force trained flag if models exist
        logging.info("BIT InjectionDetector initialized.")

    def detect(self, tool_output: str) -> tuple[bool, float]:
        # Scan the text
        result = self.detector.scan(tool_output)
        
        is_injection = result["is_injection"]
        injection_score = result["score"] # Probability of injection
        
        # AgentDojo expects (is_injection, safety_score)
        # safety_score = 1.0 - injection_score
        safety_score = 1.0 - injection_score
        
        return is_injection, safety_score

# Monkey patch AgentPipeline.from_config to support 'bit_guard'
original_from_config = AgentPipeline.from_config


# Custom get_llm to support Azure
def get_llm_custom(provider, model, model_id, tool_delimiter):
    # Support model name override
    if os.getenv("MODEL_OVERRIDE"):
        model = os.getenv("MODEL_OVERRIDE")
        model_id = model # Ensure local provider uses this too
        logging.info(f"Overriding model name with: {model}")

    # Check for Azure
    if os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"):
        try:
            from openai import AzureOpenAI
            logging.info("Using Azure OpenAI Client")
            client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
            )
            # OpenAILLM expects a client and model name. 
            # For Azure, model should match deployment name.
            return OpenAILLM(client, model)
        except ImportError:
            logging.warning("Could not import AzureOpenAI, falling back to standard config")
            pass

    # Use default get_llm
    return get_llm(provider, model, model_id, tool_delimiter)

def patched_from_config(cls, config):
    # Common Setup logic reused for custom defenses
    # Resolve ModelsEnum safely (handling name vs value)
    try:
        model_enum = ModelsEnum(config.llm)
    except ValueError:
        try:
            model_enum = ModelsEnum[config.llm]
        except KeyError:
            # Fallback or let it fail naturally in get()
            logging.warning(f"Could not resolve model {config.llm} to ModelsEnum")
            model_enum = None

    provider = MODEL_PROVIDERS.get(model_enum, "openai") if model_enum else "openai"

    llm = (
        get_llm_custom(
            provider, 
            config.llm, 
            config.model_id, 
            config.tool_delimiter
        )
        if isinstance(config.llm, str)
        else config.llm
    )
    llm_name = config.llm if isinstance(config.llm, str) else llm.name
    
    system_message_component = SystemMessage(config.system_message)
    init_query_component = InitQuery()
    
    if config.tool_output_format == "json":
        tool_output_formatter = partial(tool_result_to_str, dump_fn=json.dumps)
    else:
        tool_output_formatter = tool_result_to_str

    # CUSTOM DEFENSE: bit_guard
    if config.defense == "bit_guard":
        tools_loop = ToolsExecutionLoop(
            [
                ToolsExecutor(tool_output_formatter),
                BitGuardDetector(
                    mode="message", 
                    raise_on_injection=False,
                    threshold=0.764
                ),
                llm,
            ]
        )

        pipeline = cls(
            [
                system_message_component,
                init_query_component,
                llm,
                tools_loop,
            ]
        )
        pipeline.name = f"{llm_name}-{config.defense}"
        return pipeline
    
    # PATCHED DEFENSE: tool_filter (Allowing Non-OpenAILLM subclasses e.g. LocalLLM)
    if config.defense == "tool_filter":
        tools_loop = ToolsExecutionLoop([ToolsExecutor(tool_output_formatter), llm])
        # REMOVED check: if not isinstance(llm, OpenAILLM): ...
        
        # We assume llm has a .client attribute (LocalLLM and OpenAILLM both do)
        client = getattr(llm, "client", None)
        if not client:
             # Fallback if no client found
             raise ValueError("LLM must have a client attribute for tool_filter")

        pipeline = cls(
            [
                system_message_component,
                init_query_component,
                OpenAILLMToolFilter(TOOL_FILTER_PROMPT, client, llm_name),
                llm,
                tools_loop,
            ]
        )
        pipeline.name = f"{llm_name}-{config.defense}"
        return pipeline
    
    return original_from_config(config)


# Apply patch to AgentPipeline
AgentPipeline.from_config = classmethod(patched_from_config)
logging.info("Patched AgentPipeline to support 'bit_guard'")

# Check and Add 'bit_guard' defense
if "bit_guard" not in DEFENSES:
    DEFENSES.append("bit_guard")

# GLOBAL PATCh: Replace get_llm in the module so it is used by original_from_config too
import agentdojo.agent_pipeline.agent_pipeline
agentdojo.agent_pipeline.agent_pipeline.get_llm = get_llm_custom
logging.info("Globally patched get_llm to support Azure OpenAI")
