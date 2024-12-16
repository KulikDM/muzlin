import os
from typing import Any, List, Optional

from pydantic.v1 import PrivateAttr

from muzlin.encoders import BaseEncoder
from muzlin.utils.defaults import EncoderDefault

# Code adapted from https://github.com/aurelio-labs/semantic-router/blob/main/semantic_router/encoders/cohere.py


class VoyageAIEncoder(BaseEncoder):
    _client: Any = PrivateAttr()
    _embed_type: Any = PrivateAttr()
    type: str = 'voyageai'
    input_type: Optional[str] = 'document'

    def __init__(
        self,
        name: Optional[str] = None,
        voyage_api_key: Optional[str] = None,
        input_type: Optional[str] = 'document',
    ):
        if name is None:
            name = EncoderDefault.VOYAGE.value['embedding_model']
        super().__init__(
            name=name,
            input_type=input_type,  # type: ignore
        )
        self.input_type = input_type
        self._client = self._initialize_client(voyage_api_key)

    def _initialize_client(self, voyage_api_key: Optional[str] = None):
        """Initializes the Voyage client.

        :param voyage_api_key: The API key for the VoyageAI client, can also
        be set via the VOYAGE_API_KEY environment variable.

        :return: An instance of the VoyageAI client.
        """
        try:
            import voyageai
        except ImportError:
            raise ImportError(
                'Please install VoyageAI to use VoyageEncoder. '
                'You can install it with: '
                '`pip install voyageai`'
            )
        voyageai_api_key = voyage_api_key or os.getenv('VOYAGE_API_KEY')
        if voyageai_api_key is None:
            raise ValueError("VoyageAI API key cannot be 'None'.")
        try:
            client = voyageai.Client(voyage_api_key)
        except Exception as e:
            raise ValueError(
                f"VoyageAI API client failed to initialize. Error: {e}"
            ) from e
        return client

    def __call__(self, docs: List[str]) -> List[List[float]]:
        if self._client is None:
            raise ValueError('VoyageAI client is not initialized.')
        try:
            embeds = self._client.embed(
                texts=docs, input_type=self.input_type, model=self.name
            )
            return embeds.embeddings
        except Exception as e:
            raise ValueError(f"Voyage API call failed. Error: {e}") from e
