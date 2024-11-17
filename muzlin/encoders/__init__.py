import apipkg

# Implement lazy loading
apipkg.initpkg(__name__, {
    'BaseEncoder': 'muzlin.encoders.base:BaseEncoder',
    'AzureOpenAIEncoder': 'muzlin.encoders.zure:AzureOpenAIEncoder',
    'BedrockEncoder': 'muzlin.encoders.bedrock:BedrockEncoder',
    'CohereEncoder': 'muzlin.encoders.cohere:CohereEncoder',
    'FastEmbedEncoder': 'muzlin.encoders.fastembed:FastEmbedEncoder',
    'GoogleEncoder': 'muzlin.encoders.google:GoogleEncoder',
    'HuggingFaceEncoder': 'muzlin.encoders.huggingface:HuggingFaceEncoder',
    'HFEndpointEncoder': 'muzlin.encoders.huggingface:HFEndpointEncoder',
    'VoyageAIEncoder': 'muzlin.encoders.voyageai:VoyageAIEncoder',
    'MistralEncoder': 'muzlin.encoders.mistral:MistralEncoder',
    'OpenAIEncoder': 'muzlin.encoders.openai:OpenAIEncoder',
})
