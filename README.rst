.. image:: https://raw.githubusercontent.com/KulikDM/muzlin/main/images/Logo.png
   :target: https://raw.githubusercontent.com/KulikDM/muzlin/main/images/Logo.png
   :alt: Muzlin

*When a filter cloth 🏳️ is needed rather than a simple RAG 🏴‍☠*

**Deployment, Stats, & License**

|badge_pypi| |badge_stars| |badge_downloads| |badge_versions|
|badge_licence|

.. |badge_pypi| image:: https://img.shields.io/pypi/v/muzlin.svg?color=brightgreen&logo=pypi&logoColor=white
   :alt: PyPI version
   :target: https://pypi.org/project/muzlin/

.. |badge_stars| image:: https://img.shields.io/github/stars/KulikDM/muzlin.svg?logo=github&logoColor=white&style=flat
   :alt: GitHub stars
   :target: https://github.com/KulikDM/muzlin/stargazers

.. |badge_downloads| image:: https://img.shields.io/badge/dynamic/xml?url=https%3A%2F%2Fstatic.pepy.tech%2Fbadge%2Fmuzlin&query=%2F%2F*%5Blocal-name()%20%3D%20%27text%27%5D%5Blast()%5D&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyBzdHlsZT0iZW5hYmxlLWJhY2tncm91bmQ6bmV3IDAgMCAyNCAyNDsiIHZlcnNpb249IjEuMSIgdmlld0JveD0iMCAwIDI0IDI0IiB4bWw6c3BhY2U9InByZXNlcnZlIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHhtbG5zOnhsaW5rPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hsaW5rIj48ZyBpZD0iaW5mbyIvPjxnIGlkPSJpY29ucyI%2BPGcgaWQ9InNhdmUiPjxwYXRoIGQ9Ik0xMS4yLDE2LjZjMC40LDAuNSwxLjIsMC41LDEuNiwwbDYtNi4zQzE5LjMsOS44LDE4LjgsOSwxOCw5aC00YzAsMCwwLjItNC42LDAtN2MtMC4xLTEuMS0wLjktMi0yLTJjLTEuMSwwLTEuOSwwLjktMiwyICAgIGMtMC4yLDIuMywwLDcsMCw3SDZjLTAuOCwwLTEuMywwLjgtMC44LDEuNEwxMS4yLDE2LjZ6IiBmaWxsPSIjZWJlYmViIi8%2BPHBhdGggZD0iTTE5LDE5SDVjLTEuMSwwLTIsMC45LTIsMnYwYzAsMC42LDAuNCwxLDEsMWgxNmMwLjYsMCwxLTAuNCwxLTF2MEMyMSwxOS45LDIwLjEsMTksMTksMTl6IiBmaWxsPSIjZWJlYmViIi8%2BPC9nPjwvZz48L3N2Zz4%3D&label=downloads
   :alt: Downloads
   :target: https://pepy.tech/project/muzlin

.. |badge_versions| image:: https://img.shields.io/pypi/pyversions/muzlin.svg?logo=python&logoColor=white
   :alt: Python versions
   :target: https://pypi.org/project/muzlin/

.. |badge_licence| image:: https://img.shields.io/github/license/KulikDM/muzlin.svg?logo=data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjMyIiBpZD0iaWNvbiIgdmlld0JveD0iMCAwIDMyIDMyIiB3aWR0aD0iMzIiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PGRlZnMgZmlsbD0iI2ViZjJlZSI+PHN0eWxlPgogICAgICAuY2xzLTEgewogICAgICAgIGZpbGw6IG5vbmU7CiAgICAgIH0KICAgIDwvc3R5bGU+PC9kZWZzPjxyZWN0IGhlaWdodD0iMiIgd2lkdGg9IjEyIiB4PSI4IiB5PSI2IiBmaWxsPSIjZWJmMmVlIi8+PHJlY3QgaGVpZ2h0PSIyIiB3aWR0aD0iMTIiIHg9IjgiIHk9IjEwIiBmaWxsPSIjZWJmMmVlIi8+PHJlY3QgaGVpZ2h0PSIyIiB3aWR0aD0iNiIgeD0iOCIgeT0iMTQiIGZpbGw9IiNlYmYyZWUiLz48cmVjdCBoZWlnaHQ9IjIiIHdpZHRoPSI0IiB4PSI4IiB5PSIyNCIgZmlsbD0iI2ViZjJlZSIvPjxwYXRoIGQ9Ik0yOS43MDcsMTkuMjkzbC0zLTNhLjk5OTQuOTk5NCwwLDAsMC0xLjQxNCwwTDE2LDI1LjU4NTlWMzBoNC40MTQxbDkuMjkyOS05LjI5M0EuOTk5NC45OTk0LDAsMCwwLDI5LjcwNywxOS4yOTNaTTE5LjU4NTksMjhIMThWMjYuNDE0MWw1LTVMMjQuNTg1OSwyM1pNMjYsMjEuNTg1OSwyNC40MTQxLDIwLDI2LDE4LjQxNDEsMjcuNTg1OSwyMFoiIGZpbGw9IiNlYmYyZWUiLz48cGF0aCBkPSJNMTIsMzBINmEyLjAwMjEsMi4wMDIxLDAsMCwxLTItMlY0QTIuMDAyMSwyLjAwMjEsMCwwLDEsNiwySDIyYTIuMDAyMSwyLjAwMjEsMCwwLDEsMiwyVjE0SDIyVjRINlYyOGg2WiIgZmlsbD0iI2ViZjJlZSIvPjxyZWN0IGNsYXNzPSJjbHMtMSIgZGF0YS1uYW1lPSImbHQ7VHJhbnNwYXJlbnQgUmVjdGFuZ2xlJmd0OyIgaGVpZ2h0PSIzMiIgaWQ9Il9UcmFuc3BhcmVudF9SZWN0YW5nbGVfIiB3aWR0aD0iMzIiIGZpbGw9IiNlYmYyZWUiLz48L3N2Zz4=
   :alt: License
   :target: https://github.com/KulikDM/muzlin/blob/master/LICENSE

----

#############
 What is it?
#############

Muzlin merges classical ML with advanced generative AI to efficiently
filter text in the context of NLP and LLMs. It answers key questions in
semantic-based workflows, such as:

-  Does a RAG/GraphRAG have the right context to answer a question?

-  Is the topk retrieved context too dense/sparse?

-  Does the generated response hallucinate or deviate from the provided
   context?

-  Should new extracted text be added to an existing RAG?

-  Can we detect inliers and outliers in collections of text embeddings
   (e.g. context, user question and answers, synthetic generated data,
   etc...)?

**Note:** While production-ready, Muzlin is still evolving and subject
to significant changes!

############
 Quickstart
############

#. **Install** Muzlin using pip:

   .. code:: bash

      pip install muzlin

#. **Create text embeddings** with a pre-trained model:

   .. code:: python

      import numpy as np
      from muzlin.encoders import HuggingFaceEncoder # Ensure torch and transformers are installed

      encoder = HuggingFaceEncoder()
      vectors = encoder(texts)  # texts is a list of strings
      vectors = np.array(vectors)
      np.save('vectors', vectors)

#. **Build an anomaly detection model** for filtering:

   .. code:: python

      from muzlin.anomaly import OutlierDetector
      from pyod.models.pca import PCA

      vectors = np.load('vectors.npy')  # Load pre-saved vectors

      od = PCA(contamination=0.02)

      clf = OutlierDetector(mlflow=False, detector=od) # Saves joblib moddel
      clf.fit(vectors)

#. **Filter new text** using the trained model:

   .. code:: python

      from muzlin.anomaly import OutlierDetector
      from muzlin.encoders import HuggingFaceEncoder
      import numpy as np

      clf = OutlierDetector(model='outlier_detector.pkl')  # Load the model
      encoder = HuggingFaceEncoder()

      vector = encoder(['Who was the first man to walk on the moon?'])
      vector = np.array(vector).reshape(1, -1)

      label = clf.predict(vector)

##############
 Integrations
##############

Muzlin integrates with a wide array of libraries for anomaly detection,
vector encoding, and graph-based setups.

+-----------------------------------+-------------------------+----------------------+
| **Anomaly Detection**             | **Encoders**            | **Vector Index**     |
+===================================+=========================+======================+
| -  Scikit-Learn                   | -  HuggingFace          | -  LangChain         |
| -  PyOD (vector)                  | -  OpenAI               | -  LlamaIndex        |
| -  PyGOD (graph)                  | -  Cohere               |                      |
| -  PyThresh (thresholding)        | -  Azure                |                      |
|                                   | -  Google               |                      |
|                                   | -  Amazon Bedrock       |                      |
|                                   | -  Fastembed            |                      |
+-----------------------------------+-------------------------+----------------------+

**Simple Schematic Implementation**

.. image:: https://raw.githubusercontent.com/KulikDM/muzlin/main/images/Simple_Example.png
   :target: https://raw.githubusercontent.com/KulikDM/muzlin/main/images/Simple_Example.png
   :alt: Muzlin Pipeline

----

###########
 Resources
###########

**Example Notebooks**

+-------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------+
| Notebook                                                                                                          | Description                                                                 |
+===================================================================================================================+=============================================================================+
| `Introduction <https://github.com/KulikDM/muzlin/blob/main/examples/00_Introduction.ipynb>`_                      | Basic semantic vector-based outlier detection                               |
+-------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------+
| `Optimal Threshold <https://github.com/KulikDM/muzlin/blob/main/examples/01_Threshold_Optimization.ipynb>`_       | Selecting optimal thresholds using various methods                          |
+-------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------+
| `Cluster-Based Filtering <https://github.com/KulikDM/muzlin/blob/main/examples/02_Cluster_Filtering.ipynb>`_      | Cluster-based filtering for question answering                              |
+-------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------+
| `Graph-Based Filtering <https://github.com/KulikDM/muzlin/blob/main/examples/03_Graph_Filtering.ipynb>`_          | Using graph-based anomaly detection for semantic graphs like GraphRAG       |
+-------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------+

############
 What Else?
############

Looking for more? Check out other useful libraries like `Semantic Router
<https://github.com/aurelio-labs/semantic-router>`_, `CRAG
<https://github.com/HuskyInSalt/CRAG>`_, and `Scikit-LLM
<https://github.com/iryna-kondr/scikit-llm>`_

----

##############
 Contributing
##############

**Muzlin is still evolving!** At the moment their are major changes
being done and the structure of Muzlin is still being refined. For now,
please leave a bug report and potential new code for any fixes or
improvements. You will be added as a co-author if it is implemented.

Once this phase has been completed then ->

Anyone is welcome to contribute to Muzlin:

-  Please share your ideas and ask questions by opening an issue.

-  To contribute, first check the Issue list for the "help wanted" tag
   and comment on the one that you are interested in. The issue will
   then be assigned to you.

-  If the bug, feature, or documentation change is novel (not in the
   Issue list), you can either log a new issue or create a pull request
   for the new changes.

-  To start, fork the **dev branch** and add your
   improvement/modification/fix.

-  To make sure the code has the same style and standard, please refer
   to detector.py for example.

-  Create a pull request to the **dev branch** and follow the pull
   request template `PR template
   <https://github.com/KulikDM/muzlin/blob/main/.github/PULL_REQUEST_TEMPLATE.md>`_

-  Please make sure that all code changes are accompanied with proper
   new/updated test functions. Automatic tests will be triggered. Before
   the pull request can be merged, make sure that all the tests pass.
