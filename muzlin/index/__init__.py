import apipkg
from typing import TYPE_CHECKING

if not TYPE_CHECKING:
    # Lazy load the package using apipkg
    apipkg.initpkg(__name__, {
        'BaseIndex': 'muzlin.index.base:BaseIndex',
        'LangchainIndex': 'muzlin.index.langchain:LangchainIndex',
        'LlamaIndex': 'muzlin.index.llama_index:LlamaIndex',
    })

else:
    # Direct imports for type checking and static analysis
    from muzlin.index.base import BaseIndex
    from muzlin.index.langchain import LangchainIndex
    from muzlin.index.llama_index import LlamaIndex
