from typing import List, Optional

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from pydantic import Field

from muzlin.index import BaseIndex
from muzlin.utils.logger import logger


class LlamaIndex(BaseIndex):
    r"""LlamaIndex class for a Llamaindex vector based index.

    This class is used to standardize vector based Llamaindex Indexes for use in Muzlin.

    Args:
        index (object): A Llamaindex vector index e.g. VectorStoreIndex. Defaults to None.
        top_k (int): Number of documents to retrieve. Defaults to 10.

    Attributes:
        retriever (object): The initialized retiever.

    """

    index: VectorStoreIndex = None
    top_k: Optional[int] = 10
    retriever: BaseRetriever = Field(default=None, exclude=True)

    def __init__(
        self, **data
    ):
        super().__init__(**data)

        if self.index is None:
            raise ValueError('LLamaindex Index is required')

        if self.top_k < 5:
            logger.warning(
                'Using less than 5 top k results may yeild suboptimal results'
            )
        if self.top_k < 1:
            raise ValueError(f"top_k needs to be >= 1, but was: {self.top_k}.")

        self.retriever: BaseRetriever = self.index.as_retriever(
            search_kwargs={'k': self.top_k})

    def __call__(self, query: str) -> List[str]:

        nodes = self.retriever.retrieve(query)
        documents = [doc.dict()['node']['text'] for doc in nodes]

        return documents
