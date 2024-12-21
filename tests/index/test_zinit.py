import sys
from importlib import reload
from unittest.mock import patch

import muzlin.index as det


class TestIndexInit:
    def test_lazy_loading(self):

        with patch('typing.TYPE_CHECKING', False), \
                patch('apipkg.initpkg') as mock_initpkg:

            reload(det)

            # Check if apipkg.initpkg was called with the correct arguments
            mock_initpkg.assert_called_once_with('muzlin.index', {
                'BaseIndex': 'muzlin.index.base:BaseIndex',
                'LangchainIndex': 'muzlin.index.langchain:LangchainIndex',
                'LlamaIndex': 'muzlin.index.llama_index:LlamaIndex',
            })

    def test_type_checking_imports(self):

        with patch('typing.TYPE_CHECKING', True), \
                patch('muzlin.index.base.BaseIndex', create=True), \
                patch('muzlin.index.langchain.LangchainIndex', create=True), \
                patch('muzlin.index.llama_index.LlamaIndex', create=True):

            reload(det)

            # Ensure lazy loading is skipped
            assert 'muzlin.index.base' in sys.modules
            assert 'muzlin.index.langchain' in sys.modules
            assert 'muzlin.index.llama_index' in sys.modules
