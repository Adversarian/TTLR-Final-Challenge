from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.protocol import apply_protocol_guards


def test_ping_guard():
    response = apply_protocol_guards("ping")
    assert response is not None
    assert response.message == "pong"
    assert response.base_random_keys is None
    assert response.member_random_keys is None


def test_base_random_key_guard():
    response = apply_protocol_guards("return base random key: abc")
    assert response is not None
    assert response.message is None
    assert response.base_random_keys == ["abc"]


def test_member_random_key_guard():
    response = apply_protocol_guards("return member random key: xyz")
    assert response is not None
    assert response.member_random_keys == ["xyz"]
