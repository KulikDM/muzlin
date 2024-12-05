import pytest
from unittest.mock import Mock, MagicMock
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever

from muzlin.index import BaseIndex, LangchainIndex


class MockVectorStore(VectorStore):
    """A mock class that mimics VectorStore with abstract methods implemented."""

    def add_texts(self, texts, **kwargs):
        """Mock implementation of add_texts."""
        return None

    @classmethod
    def from_texts(cls, texts, **kwargs):
        """Mock implementation of from_texts."""
        return cls()

    def similarity_search(self, query, **kwargs):
        """Mock implementation of similarity_search."""
        return [
            Mock(page_content="Document 1"),
            Mock(page_content="Document 2"),
        ]

    def as_retriever(self, **kwargs):
        """Mock implementation of as_retriever."""
        retriever = MagicMock(spec=VectorStoreRetriever)
        # Return a list of mock documents with `page_content` attributes
        retriever.invoke.return_value = [
            Mock(page_content="Document 1"),
            Mock(page_content="Document 2"),
        ]
        return retriever

@pytest.fixture
def mock_vector_store():
    """Provide a mock VectorStore for testing."""
    mock_store = MockVectorStore()
    mock_store.as_retriever = MagicMock()
    mock_store.as_retriever.return_value = MagicMock(spec=VectorStoreRetriever)
    mock_store.as_retriever.return_value.invoke.return_value = [
        Mock(page_content="Document 1"),
        Mock(page_content="Document 2"),
    ]
    return mock_store

class TestLangchainIndex:
    def test_langchain_index_initialization(self, mock_vector_store):
        """Test initialization of LangchainIndex."""
        
        index = LangchainIndex(index=mock_vector_store, top_k=5)

        assert index.top_k == 5
        assert isinstance(index.retriever, VectorStoreRetriever)


    def test_langchain_index_call(self, mock_vector_store):
        """Test calling the LangchainIndex."""
        
        index = LangchainIndex(index=mock_vector_store, top_k=2)
        query = "Find documents about LangChain."

        result = index(query)

        assert result == ["Document 1", "Document 2"]
        mock_vector_store.as_retriever().invoke.assert_called_once_with(query)


    def test_langchain_index_top_k_validation(self, mock_vector_store):
        """Test top_k validation logic."""
        
        with pytest.raises(ValueError, match="top_k needs to be >= 1"):
            LangchainIndex(index=mock_vector_store, top_k=0)


    def test_langchain_index_missing_index(self):
        """Test ValueError raised when index is None."""
        
        with pytest.raises(ValueError, match="Langchain Index is required"):
            LangchainIndex(index=None, top_k=5)
