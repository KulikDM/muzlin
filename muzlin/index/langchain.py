from typing import List, Optional, Union

from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from pydantic import Field

from muzlin.index import BaseIndex
from muzlin.utils.logger import logger


class LangchainIndex(BaseIndex):
    r"""LangchainIndex class for a LangChain vector based index.

    This class is used to standardize vector based LangChain Indexes for use in Muzlin.

    Args:
        index (object): A LangChain vector index e.g. FAISS. Defaults to None.
        top_k (int): Number of documents to retrieve. Defaults to 10.

    Attributes:
        retriever (object): The initialized retiever.

    """

    index: Union[VectorStore, VectorStoreRetriever] = None
    top_k: Optional[int] = 10
    retriever: VectorStoreRetriever = Field(default=None, exclude=True)

    def __init__(
        self, **data
    ):
        super().__init__(**data)

        if self.index is None:
            raise ValueError('Langchain Index is required')

        if self.top_k < 5:
            logger.warning(
                'Using less than 5 top k results may yeild suboptimal results'
            )
        if self.top_k < 1:
            raise ValueError(f"top_k needs to be >= 1, but was: {self.top_k}.")

        self.retriever:  VectorStoreRetriever = self.index.as_retriever(search_kwargs={
                                                                        'k': self.top_k})

    def __call__(self, query: str) -> List[str]:

        docs_dict = self.retriever.invoke(query)
        documents = [doc.page_content for doc in docs_dict]

        return documents
