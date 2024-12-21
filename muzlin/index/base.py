from typing import Any, List, Union

from pydantic import BaseModel


class BaseIndex(BaseModel):

    index: Union[Any, None]

    class Config:
        arbitrary_types_allowed = True

    def __call__(self, query: str) -> List[str]:
        raise NotImplementedError('Subclasses must implement this method')
