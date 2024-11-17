from enum import Enum

from pydantic.v1 import BaseModel


class EncoderType(Enum):
    AZURE = 'azure'
    COHERE = 'cohere'
    OPENAI = 'openai'
    FASTEMBED = 'fastembed'
    HUGGINGFACE = 'huggingface'
    GOOGLE = 'google'
    BEDROCK = 'bedrock'
    VOYAGE = 'voyage'
    MISTRAL = 'mistral'


class EncoderInfo(BaseModel):
    name: str
    token_limit: int


class IndexType(Enum):
    LANGCHAIN = 'langchain'
    LLAMAINDEX = 'llamaindex'
