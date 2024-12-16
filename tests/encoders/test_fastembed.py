from muzlin.encoders import FastEmbedEncoder

# Code adapted from https://github.com/aurelio-labs/semantic-router/blob/main/tests/unit/encoders/test_fastembed.py


class TestFastEmbedEncoder:
    def test_fastembed_encoder(self):
        encode = FastEmbedEncoder()
        test_docs = ['This is a test', 'This is another test']
        embeddings = encode(test_docs)
        assert isinstance(embeddings, list)
