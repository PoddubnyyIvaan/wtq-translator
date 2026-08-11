from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderCapabilities:
    structured_output: bool = False
    json_schema: bool = False
    seed: bool = False
    streaming: bool = False
