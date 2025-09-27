from ai_dev_agent.core.utils.state import StateStore


def test_state_store_roundtrip(tmp_path):
    path = tmp_path / "state.json"
    store = StateStore(path)
    store.save({"a": 1})
    assert store.load()["a"] == 1
    store.update(b=2)
    data = store.load()
    assert data["b"] == 2
