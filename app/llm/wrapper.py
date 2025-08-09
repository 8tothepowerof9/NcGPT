import dspy
from contextlib import contextmanager
from ..models import Provider, LLMConfig
from ..core.config import MODEL_NAME, LLM_PROVIDER, API_KEY


default_config = LLMConfig(
    provider=LLM_PROVIDER,
    model=MODEL_NAME,
    api_key=API_KEY,
)


class DspyLLM:
    """
    Wrapper for dspy.LM to provide a unified interface for different LLM providers.
    Keep graphs/programs provider-agnostic.
    """

    OLLAMA_API_BASE = "http://localhost:11434"

    def __init__(self, config: LLMConfig = default_config):
        """
        Initializes the DspyLLM with the given configuration.

        Args:
            config (LLMConfig): Configuration for the LLM, including provider, model, API key, etc.
        """
        self.config = config

    def build(self) -> dspy.LM:
        """
        Builds the dspy.LM instance based on the configuration.


        Returns:
            dspy.LM: dspy.LM instance configured with the specified provider and model.
        """
        # Get prefix
        prefix = self.config.provider.value
        name = f"{prefix}/{self.config.model}"

        # Use default api_base from Provider if not set in config
        api_base = self.config.api_base or self.config.provider.default_api_base

        return dspy.LM(
            name, api_base=api_base, api_key=self.config.api_key, **self.config.extra
        )

    def configure_global(self) -> "DspyLLM":
        """
        Configures the global dspy.LM instance with the current configuration.

        This method sets the global dspy.LM instance to the one built from the current configuration.

        Returns:
            DspyLLM: The current DspyLLM instance for method chaining.
        """
        dspy.configure(lm=self.build())
        return self

    @contextmanager
    def context(self):
        """
        Context manager to temporarily set the dspy.LM instance for the duration of the block.

        This allows you to run code with a specific dspy.LM instance without affecting the global state.
        """
        with dspy.context(lm=self.build()):
            yield
