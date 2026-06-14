from google import genai
import os

class GeminiClient:
    """Class responsible for communicating with the Gemini model via Google AI Studio"""
    
    def __init__(self, api_key, model_name=None):
        # Initialize the client with the new SDK
        self.client = genai.Client(api_key=api_key)
        # Allow passing the model explicitly, fallback to environment or default
        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.last_usage = None

    def _normalize_usage(self, usage):
        if usage is None:
            return None

        if isinstance(usage, dict):
            return usage

        normalized = {}
        for field_name in (
            "prompt_token_count",
            "candidates_token_count",
            "cached_content_token_count",
            "total_token_count",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
        ):
            if hasattr(usage, field_name):
                normalized[field_name] = getattr(usage, field_name)

        return normalized or None
        
    def generate(self, prompt):
        """Send a prompt to the model and receive a response"""
        try:
            # Send Prompt and receive the reply
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            usage = getattr(response, "usage_metadata", None) or getattr(response, "usage", None)
            self.last_usage = self._normalize_usage(usage)
            return response.text
        except Exception as e:
            self.last_usage = None
            print(f"An error occurred in Gemini API: {e}")
            return None
