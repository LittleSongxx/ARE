from __future__ import annotations

from collections import defaultdict, deque
import heapq
from typing import Dict, Iterable, List, Tuple

import torch

from ac_pbgrl.state import ExplorationState
from ac_pbgrl.utils import stable_coordinate_id


def _bfs_distance(adjacency: torch.Tensor, start: int, valid: set[int]) -> dict[int, int]:
    distance = {start: 0}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        neighbors = torch.nonzero(adjacency[node], as_tuple=False).flatten().tolist()
        for neighbor in neighbors:
            if neighbor in valid and neighbor not in distance:
                distance[neighbor] = distance[node] + 1
                queue.append(neighbor)
    return distance


def _a_star_path(
    adjacency: torch.Tensor,
    xy: torch.Tensor,
    start: int,
    goal: int,
    valid: set[int],
) -> list[int]:
    """A* on the original executable graph; used only for context skeletons."""

    if start == goal:
        return [start]
    coordinates = xy.detach().cpu()
    frontier: list[tuple[float, float, int]] = [(0.0, 0.0, start)]
    parent: dict[int, int] = {}
    cost = {start: 0.0}
    while frontier:
        _, current_cost, node = heapq.heappop(frontier)
        if current_cost > cost.get(node, float("inf")):
            continue
        if node == goal:
            path = [goal]
            while path[-1] != start:
                path.append(parent[path[-1]])
            return list(reversed(path))
        neighbors = torch.nonzero(adjacency[node], as_tuple=False).flatten().tolist()
        for neighbor in neighbors:
            if neighbor not in valid or neighbor == node:
                continue
            edge_cost = float(torch.linalg.vector_norm(coordinates[node] - coordinates[neighbor]))
            candidate = current_cost + max(edge_cost, 1.0e-6)
            if candidate >= cost.get(neighbor, float("inf")):
                continue
            cost[neighbor] = candidate
            parent[neighbor] = node
            heuristic = float(torch.linalg.vector_norm(coordinates[neighbor] - coordinates[goal]))
            heapq.heappush(frontier, (candidate + heuristic, candidate, neighbor))
    return []


def action_preserving_context(
    state: ExplorationState,
    *,
    local_budget: int = 192,
    region_budget: int = 32,
    region_size_m: float = 16.0,
    utility_feature_index: int = 2,
) -> ExplorationState:
    """Compact a graph while preserving every action slot and its node identity.

    Local nodes remain original nodes. Only nodes outside the local budget may be
    pooled into context-only region tokens. Candidate slots are remapped to the
    retained local indices without changing their order.
    """

    state.validate()
    device = state.node_features.device
    batch_size, _, feature_dim = state.node_features.shape
    total_budget = local_budget + region_budget
    output_features = state.node_features.new_zeros((batch_size, total_budget, feature_dim))
    output_xy = state.node_xy.new_zeros((batch_size, total_budget, 2))
    output_mask = torch.zeros((batch_size, total_budget), dtype=torch.bool, device=device)
    output_adjacency = torch.zeros((batch_size, total_budget, total_budget), dtype=torch.bool, device=device)
    output_ids = torch.zeros((batch_size, total_budget), dtype=torch.long, device=device)
    output_current = torch.zeros((batch_size,), dtype=torch.long, device=device)
    output_candidates = torch.zeros_like(state.candidate_indices)

    compaction_metadata: list[dict] = []
    for batch_index in range(batch_size):
        valid_nodes = set(torch.nonzero(state.node_mask[batch_index], as_tuple=False).flatten().tolist())
        current = int(state.current_index[batch_index])
        candidate_slots = torch.nonzero(state.candidate_mask[batch_index], as_tuple=False).flatten().tolist()
        candidates = [int(state.candidate_indices[batch_index, slot]) for slot in candidate_slots]
        protected = list(dict.fromkeys([current] + candidates))
        if len(protected) > local_budget:
            raise ValueError(
                f"local budget {local_budget} cannot preserve current node and {len(candidates)} candidates"
            )

        distances = _bfs_distance(state.adjacency[batch_index].bool(), current, valid_nodes)
        ranked = []
        for node in valid_nodes:
            if node in protected:
                continue
            utility = 0.0
            if utility_feature_index < feature_dim:
                utility = float(state.node_features[batch_index, node, utility_feature_index].detach().cpu())
            graph_distance = distances.get(node, 10**6)
            score = (graph_distance < 10**6, -graph_distance, utility, -node)
            ranked.append((score, node))
        ranked.sort(reverse=True)
        local_nodes = protected + [node for _, node in ranked[: max(0, local_budget - len(protected))]]
        local_nodes = list(dict.fromkeys(local_nodes))
        local_set = set(local_nodes)
        remote_nodes = sorted(valid_nodes - local_set)
        local_lookup = {old: new for new, old in enumerate(local_nodes)}

        region_groups: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        for old_index in remote_nodes:
            x, y = state.node_xy[batch_index, old_index].detach().cpu().tolist()
            region_groups[(int(x // region_size_m), int(y // region_size_m))].append(old_index)
        ranked_regions = []
        for key, members in region_groups.items():
            utility = 0.0
            if utility_feature_index < feature_dim:
                utility = float(
                    state.node_features[batch_index, members, utility_feature_index].clamp_min(0).sum().detach().cpu()
                )
            ranked_regions.append(((utility, len(members), -abs(key[0]) - abs(key[1])), key, members))
        ranked_regions.sort(reverse=True)
        selected_regions = ranked_regions[:region_budget]

        local_count = len(local_nodes)
        if local_nodes:
            local_tensor = torch.tensor(local_nodes, dtype=torch.long, device=device)
            output_features[batch_index, :local_count] = state.node_features[batch_index].index_select(0, local_tensor)
            output_xy[batch_index, :local_count] = state.node_xy[batch_index].index_select(0, local_tensor)
            output_ids[batch_index, :local_count] = state.stable_ids[batch_index].index_select(0, local_tensor)
            local_adjacency = state.adjacency[batch_index].index_select(0, local_tensor).index_select(1, local_tensor)
            output_adjacency[batch_index, :local_count, :local_count] = local_adjacency.bool()
            output_mask[batch_index, :local_count] = True

        old_to_region: dict[int, int] = {}
        region_members: list[list[int]] = []
        region_representatives: list[int] = []
        for offset, (_, region_key, members) in enumerate(selected_regions):
            new_index = local_count + offset
            member_tensor = torch.tensor(members, dtype=torch.long, device=device)
            output_features[batch_index, new_index] = state.node_features[batch_index].index_select(0, member_tensor).mean(0)
            output_xy[batch_index, new_index] = state.node_xy[batch_index].index_select(0, member_tensor).mean(0)
            center_x = (region_key[0] + 0.5) * region_size_m
            center_y = (region_key[1] + 0.5) * region_size_m
            output_ids[batch_index, new_index] = stable_coordinate_id(center_x, center_y, 0.1, namespace=1)
            output_mask[batch_index, new_index] = True
            region_members.append(members)
            region_center = output_xy[batch_index, new_index]
            representative = min(
                members,
                key=lambda member: float(
                    torch.linalg.vector_norm(state.node_xy[batch_index, member] - region_center).detach().cpu()
                ),
            )
            region_representatives.append(representative)
            for member in members:
                old_to_region[member] = new_index

        # Collapse original connectivity into local-to-region and region-to-region edges.
        selected_old = set(local_nodes) | set(old_to_region)
        for old_source in selected_old:
            new_source = local_lookup.get(old_source, old_to_region.get(old_source))
            if new_source is None:
                continue
            neighbors = torch.nonzero(state.adjacency[batch_index, old_source], as_tuple=False).flatten().tolist()
            for old_target in neighbors:
                new_target = local_lookup.get(old_target, old_to_region.get(old_target))
                if new_target is not None and new_source != new_target:
                    output_adjacency[batch_index, new_source, new_target] = True
                    output_adjacency[batch_index, new_target, new_source] = True

        # Every remote context token receives a shortest-path skeleton back to
        # the real local action graph. Omitted remote nodes are contracted along
        # the path; no synthetic token is ever introduced into current_edge.
        skeleton_paths = []
        for offset, representative in enumerate(region_representatives):
            path = _a_star_path(
                state.adjacency[batch_index].bool(),
                state.node_xy[batch_index],
                representative,
                current,
                valid_nodes,
            )
            mapped = []
            for old_node in path:
                new_node = local_lookup.get(old_node, old_to_region.get(old_node))
                if new_node is not None and (not mapped or mapped[-1] != new_node):
                    mapped.append(new_node)
            for left, right in zip(mapped[:-1], mapped[1:]):
                output_adjacency[batch_index, left, right] = True
                output_adjacency[batch_index, right, left] = True
            if path:
                points = state.node_xy[batch_index, torch.tensor(path, device=device)]
                length = float(torch.linalg.vector_norm(points[1:] - points[:-1], dim=-1).sum().detach().cpu())
            else:
                length = float("inf")
            skeleton_paths.append(
                {
                    "region_token": local_count + offset,
                    "original_path": path,
                    "contracted_tokens": mapped,
                    "path_length": length,
                }
            )
        valid_count = local_count + len(selected_regions)
        diagonal = torch.arange(valid_count, device=device)
        output_adjacency[batch_index, diagonal, diagonal] = True
        output_current[batch_index] = local_lookup[current]
        for slot, old_candidate in zip(candidate_slots, candidates):
            if old_candidate not in local_lookup:
                raise AssertionError("candidate node was not preserved by graph compaction")
            output_candidates[batch_index, slot] = local_lookup[old_candidate]
        compaction_metadata.append(
            {
                "original_nodes": len(valid_nodes),
                "local_nodes": local_count,
                "region_tokens": len(selected_regions),
                "candidate_old_indices": candidates,
                "candidate_new_indices": [local_lookup[item] for item in candidates],
                "a_star_skeletons": skeleton_paths,
            }
        )

    metadata = dict(state.metadata or {})
    metadata["compaction"] = compaction_metadata
    result = ExplorationState(
        node_features=output_features,
        node_xy=output_xy,
        node_mask=output_mask,
        adjacency=output_adjacency,
        stable_ids=output_ids,
        current_index=output_current,
        candidate_indices=output_candidates,
        candidate_mask=state.candidate_mask,
        edge_features=state.edge_features,
        candidate_events=state.candidate_events,
        posterior_mean=state.posterior_mean,
        posterior_variance=state.posterior_variance,
        metadata=metadata,
    )
    return result.validate()
