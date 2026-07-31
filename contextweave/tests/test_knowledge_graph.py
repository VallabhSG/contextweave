"""Tests for knowledge-graph traversal, focused on hop-distance ranking."""

from __future__ import annotations

import pytest

from contextweave.schemas import Entity
from contextweave.storage.knowledge_graph import KnowledgeGraph
from contextweave.timeutils import utcnow


@pytest.fixture
def graph(tmp_path):
    return KnowledgeGraph(db_path=str(tmp_path / "graph.db"))


def entity(name: str) -> Entity:
    now = utcnow()
    return Entity(name=name, entity_type="topic", first_seen=now, last_seen=now)


def build_chain(graph: KnowledgeGraph) -> None:
    # A co-occurs with B in chunk1; B co-occurs with C in chunk2. So from A:
    # A is 0 hops, B is 1 hop, C is 2 hops.
    graph.add_entities([entity("A"), entity("B")], "chunk1")
    graph.add_entities([entity("B"), entity("C")], "chunk2")


class TestNeighborDistance:
    def test_distance_increases_per_hop(self, graph):
        build_chain(graph)
        dist = graph.get_neighbors_with_distance("A", hops=2)
        assert dist["A"] == 0
        assert dist["B"] == 1
        assert dist["C"] == 2

    def test_hops_bound_the_frontier(self, graph):
        build_chain(graph)
        dist = graph.get_neighbors_with_distance("A", hops=1)
        assert set(dist) == {"A", "B"}, "C is 2 hops away and must be excluded at hops=1"

    def test_unknown_entity_is_empty(self, graph):
        assert graph.get_neighbors_with_distance("nobody", hops=2) == {}


class TestConnectedChunksRanked:
    def test_chunk_distance_is_nearest_referencing_entity(self, graph):
        build_chain(graph)
        ranked = graph.get_connected_chunks_ranked("A", hops=2)
        assert ranked["chunk1"] == 0, "A directly references chunk1"
        assert ranked["chunk2"] == 1, "chunk2 is reached via 1-hop neighbour B"

    def test_get_connected_chunks_orders_nearest_first(self, graph):
        build_chain(graph)
        assert graph.get_connected_chunks("A", hops=2) == ["chunk1", "chunk2"]

    def test_unknown_entity_is_empty(self, graph):
        assert graph.get_connected_chunks_ranked("nobody", hops=2) == {}
        assert graph.get_connected_chunks("nobody", hops=2) == []
