from typing import List
from basemodel import Model


class MockModel(Model):
    @classmethod
    def supported_models(cls) -> List[str]:
        return ["mock_model"]

    def _verify_model(self):
        pass

    def set_key(self, api_key: str):
        pass

    def set_model(self, model: str):
        pass

    def get_description(self) -> str:
        return "Mock model for testing"

    def get_endpoint(self) -> str:
        return "https://mock.endpoint/"

    def get_parameters(self) -> Dict[str, str]:
        return {"param": "value"}

    def run(self, prompts: List[str]) -> List[str]:
        return ["response" for _ in prompts]

    def model_output(self, response):
        return response
