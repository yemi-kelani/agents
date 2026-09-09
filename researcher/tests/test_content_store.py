"""Spec 01: the content store keeps fetched text out of graph state."""
from __future__ import annotations

import pytest

from content_store import ContentStore


def test_source_id_is_derived_from_url_hash():
    sid = ContentStore.source_id("https://example.com/a")
    assert len(sid) == 16
    assert sid == ContentStore.source_id("https://example.com/a")


def test_different_urls_get_different_ids():
    assert ContentStore.source_id("https://a.com") != ContentStore.source_id("https://b.com")


def test_put_then_get_round_trips():
    store = ContentStore()
    store.put("s1", "hello")
    assert store.get("s1") == "hello"


def test_contains_reports_membership():
    store = ContentStore()
    store.put("s1", "hello")
    assert "s1" in store
    assert "s2" not in store


def test_get_raises_on_missing_source():
    store = ContentStore()
    with pytest.raises(KeyError):
        store.get("nope")


def test_put_overwrites_same_source_id():
    store = ContentStore()
    store.put("s1", "first")
    store.put("s1", "second")
    assert store.get("s1") == "second"
    assert len(store) == 1
