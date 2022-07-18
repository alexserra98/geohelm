from .gpt2_tokenizer import GPT2Tokenizer
from .tokenizer_service import TokenizerService


class GPTJTokenizer(GPT2Tokenizer):
    """
    The same tokenizer as GPT-2, but with an additional 143 tokens
    (source: https://huggingface.co/docs/transformers/model_doc/gptj).
    """

    # The max length of the model input. The max sequence length for GPT-J is 2048.
    MAX_SEQUENCE_LENGTH: int = 2048

    # The max request length of GPT-J is MAX_SEQUENCE_LENGTH + 1.
    MAX_REQUEST_LENGTH: int = 2049

    def __init__(self, service: TokenizerService):
        super().__init__(service)

    @property
    def max_sequence_length(self) -> int:
        """Return the max sequence length."""
        return GPTJTokenizer.MAX_SEQUENCE_LENGTH

    @property
    def max_request_length(self) -> int:
        """Return the max request length."""
        return GPTJTokenizer.MAX_REQUEST_LENGTH

    @property
    def tokenizer_name(self) -> str:
        """Name of the tokenizer to use when sending a request."""
        return "huggingface/gpt-j-6b"
