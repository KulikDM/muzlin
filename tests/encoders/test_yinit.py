import sys
from importlib import reload
from unittest.mock import patch

import muzlin.encoders as det


class TestEncoderInit:
    def test_lazy_loading(self):

        with patch('typing.TYPE_CHECKING', False), \
                patch('apipkg.initpkg') as mock_initpkg:

            reload(det)

            # Check if apipkg.initpkg was called with the correct arguments
            mock_initpkg.assert_called_once_with('muzlin.encoders', {
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

    def test_type_checking_imports(self):

        with patch('typing.TYPE_CHECKING', True), \
                patch('muzlin.encoders.base.BaseEncoder', create=True), \
                patch('muzlin.encoders.zure.AzureOpenAIEncoder', create=True), \
                patch('muzlin.encoders.bedrock.BedrockEncoder', create=True), \
                patch('muzlin.encoders.cohere.CohereEncoder', create=True), \
                patch('muzlin.encoders.fastembed.FastEmbedEncoder', create=True), \
                patch('muzlin.encoders.google.GoogleEncoder', create=True), \
                patch('muzlin.encoders.huggingface.HuggingFaceEncoder', create=True), \
                patch('muzlin.encoders.huggingface.HFEndpointEncoder', create=True), \
                patch('muzlin.encoders.mistral.MistralEncoder', create=True), \
                patch('muzlin.encoders.openai.OpenAIEncoder', create=True), \
                patch('muzlin.encoders.voyageai.VoyageAIEncoder', create=True):

            reload(det)

            # Ensure lazy loading is skipped
            assert 'muzlin.encoders.base' in sys.modules
            assert 'muzlin.encoders.zure' in sys.modules
            assert 'muzlin.encoders.bedrock' in sys.modules
            assert 'muzlin.encoders.cohere' in sys.modules
            assert 'muzlin.encoders.fastembed' in sys.modules
            assert 'muzlin.encoders.google' in sys.modules
            assert 'muzlin.encoders.huggingface' in sys.modules
            assert 'muzlin.encoders.mistral' in sys.modules
            assert 'muzlin.encoders.openai' in sys.modules
            assert 'muzlin.encoders.voyageai' in sys.modules
