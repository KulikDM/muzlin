from unittest.mock import MagicMock, Mock

import pytest
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.indices.vector_store.base import VectorStoreIndex

from muzlin.index import LlamaIndex


class MockVectorStore(VectorStoreIndex):
    """A mock class that mimics VectorStoreIndex with abstract methods implemented."""

    def __init__(self):
        self.mock_embedding = MagicMock(embed_dim=8)

    @staticmethod
    def mock_openai_key_validation():
        """Mock OpenAI API key validation."""
        validate_openai_api_key = MagicMock()
        validate_openai_api_key.return_value = None
        return validate_openai_api_key

    @staticmethod
    def mock_embedding_model():
        """Mock embedding model."""
        return MagicMock(embed_dim=8)

    def as_retriever(self, **kwargs):
        """Mock implementation of as_retriever."""
        retriever = MagicMock(spec=BaseRetriever)
        # Return a list of mock documents with `page_content` attributes
        retriever.retrieve.return_value = [
            Mock(dict=lambda: {'node': {'text': 'Document 1'}}),
            Mock(dict=lambda: {'node': {'text': 'Document 2'}}),
        ]
        return retriever


@pytest.fixture
def mock_vector_store():
    """Provide a mock VectorStoreIndex for testing."""
    mock_store = MockVectorStore()
    mock_store.as_retriever = MagicMock()
    mock_store.as_retriever.return_value = MagicMock(spec=BaseRetriever)
    mock_store.as_retriever.return_value.retrieve.return_value = [
        Mock(dict=lambda: {'node': {'text': 'Document 1'}}),
        Mock(dict=lambda: {'node': {'text': 'Document 2'}}),
    ]

    # Mock OpenAI API key validation
    mock_store.mock_openai_key_validation()

    # Mock embedding model
    mock_store.mock_embedding_model()

    return mock_store


class TestLlamaIndex:
    def test_index_initialization(self, mock_vector_store):
        """Test initialization of LlamaIndex."""

        index = LlamaIndex(index=mock_vector_store, top_k=5)

        assert index.top_k == 5
        assert isinstance(index.retriever, BaseRetriever)

    def test_index_call(self, mock_vector_store):
        """Test calling the LlamaIndex."""

        index = LlamaIndex(index=mock_vector_store, top_k=2)
        query = 'Find documents about LlamaIndex.'

        result = index(query)

        assert result == ['Document 1', 'Document 2']
        mock_vector_store.as_retriever().retrieve.assert_called_once_with(query)

    def test_index_top_k_validation(self, mock_vector_store):
        """Test top_k validation logic."""

        with pytest.raises(ValueError, match='top_k needs to be >= 1'):
            LlamaIndex(index=mock_vector_store, top_k=0)

    def test_index_missing_index(self):
        """Test ValueError raised when index is None."""

        with pytest.raises(ValueError, match='LLamaindex Index is required'):
            LlamaIndex(index=None, top_k=5)
