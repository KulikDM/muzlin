import pytest

from muzlin.encoders import BaseEncoder

# Code adapted from https://github.com/aurelio-labs/semantic-router/blob/main/tests/unit/encoders/test_base.py


class TestBaseEncoder:
    @pytest.fixture
    def base_encoder(self):
        return BaseEncoder(name="TestEncoder")

    def test_base_encoder_initialization(self, base_encoder):
        assert base_encoder.name == "TestEncoder", "Initialization of name failed"

    def test_base_encoder_call_method_not_implemented(self, base_encoder):
        with pytest.raises(NotImplementedError):
            base_encoder(["some", "texts"])