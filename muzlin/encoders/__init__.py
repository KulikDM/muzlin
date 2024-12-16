from typing import TYPE_CHECKING

import apipkg

if not TYPE_CHECKING:
    # Lazy load the package using apipkg
    apipkg.initpkg(__name__, {
        'BaseEncoder': 'muzlin.encoders.base:BaseEncoder',
        'AzureOpenAIEncoder': 'muzlin.encoders.zure:AzureOpenAIEncoder',
        'BedrockEncoder': 'muzlin.encoders.bedrock:BedrockEncoder',
        'CohereEncoder': 'muzlin.encoders.cohere:CohereEncoder',
        'FastEmbedEncoder': 'muzlin.encoders.fastembed:FastEmbedEncoder',
        'GoogleEncoder': 'muzlin.encoders.google:GoogleEncoder',
        'HuggingFaceEncoder': 'muzlin.encoders.huggingface:HuggingFaceEncoder',
        'HFEndpointEncoder': 'muzlin.encoders.huggingface:HFEndpointEncoder',
        'MistralEncoder': 'muzlin.encoders.mistral:MistralEncoder',
        'OpenAIEncoder': 'muzlin.encoders.openai:OpenAIEncoder',
        'VoyageAIEncoder': 'muzlin.encoders.voyageai:VoyageAIEncoder',
    })

else:
    # Direct imports for type checking and static analysis
    from muzlin.encoders.base import BaseEncoder
    from muzlin.encoders.bedrock import BedrockEncoder
    from muzlin.encoders.cohere import CohereEncoder
    from muzlin.encoders.fastembed import FastEmbedEncoder
    from muzlin.encoders.google import GoogleEncoder
    from muzlin.encoders.huggingface import HFEndpointEncoder, HuggingFaceEncoder
    from muzlin.encoders.mistral import MistralEncoder
    from muzlin.encoders.openai import OpenAIEncoder
    from muzlin.encoders.voyageai import VoyageAIEncoder
    from muzlin.encoders.zure import AzureOpenAIEncoder
