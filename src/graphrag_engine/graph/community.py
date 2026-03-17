from __future__ import annotations

from collections import defaultdict, deque

from graphrag_engine.common.models import EntityRecord, RelationRecord

try:
    import igraph as ig  # type: ignore
    import leidenalg  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    ig = None
    leidenalg = None


def detect_communities(entities: list[EntityRecord], relations: list[RelationRecord]) -> dict[str, int]:
    if ig is not None and leidenalg is not None and entities:
        graph = ig.Graph(directed=False)
        entity_ids = [entity.entity_id for entity in entities]
        graph.add_vertices(entity_ids)
        edges = [
            (relation.subject_entity_id, relation.object_entity_id)
            for relation in relations
            if relation.subject_entity_id in entity_ids and relation.object_entity_id in entity_ids
        ]
        if edges:
            graph.add_edges(edges)
        partition = leidenalg.find_partition(graph, leidenalg.ModularityVertexPartition)
        return {
            entity_ids[index]: community_id
            for community_id, cluster in enumerate(partition)
            for index in cluster
        }

    adjacency: dict[str, set[str]] = defaultdict(set)
    for relation in relations:
        adjacency[relation.subject_entity_id].add(relation.object_entity_id)
        adjacency[relation.object_entity_id].add(relation.subject_entity_id)

    community_by_entity: dict[str, int] = {}
    community_id = 0
    for entity in entities:
        if entity.entity_id in community_by_entity:
            continue
        queue = deque([entity.entity_id])
        while queue:
            current = queue.popleft()
            if current in community_by_entity:
                continue
            community_by_entity[current] = community_id
            queue.extend(adjacency.get(current, set()))
        community_id += 1
    return community_by_entity
