from src.core.config import config


class ModelManager:
    def __init__(self, config):
        self.config = config

    def get_tier(self, claude_model: str) -> str:
        """Return the tier name (opus/sonnet/haiku) for a Claude model."""
        if claude_model.startswith("gpt-") or claude_model.startswith("o1-"):
            return "sonnet"
        if claude_model.startswith("ep-") or claude_model.startswith("doubao-") or claude_model.startswith("deepseek-"):
            return "sonnet"

        model_lower = claude_model.lower()
        if "haiku" in model_lower:
            return "haiku"
        elif "sonnet" in model_lower:
            return "sonnet"
        elif "opus" in model_lower:
            return "opus"
        else:
            return "sonnet"

    def map_claude_model_to_openai(self, claude_model: str) -> str:
        """Map Claude model names to OpenAI model names based on tier config."""
        # If it's already an OpenAI model, return as-is
        if claude_model.startswith("gpt-") or claude_model.startswith("o1-"):
            return claude_model

        # If it's other supported models, return as-is
        if claude_model.startswith("ep-") or claude_model.startswith("doubao-") or claude_model.startswith("deepseek-"):
            return claude_model

        tier = self.get_tier(claude_model)
        provider = self.config.get_tier(tier)
        return provider.model


model_manager = ModelManager(config)
