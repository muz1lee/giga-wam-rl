from collections.abc import Sequence


def valid_window_indices(episode_lengths: Sequence[int], *, horizon: int) -> list[int]:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    indices = []
    episode_start = 0
    for length in episode_lengths:
        if length < 0:
            raise ValueError("episode lengths must be non-negative")
        indices.extend(range(episode_start, episode_start + max(length - horizon, 0)))
        episode_start += length
    return indices


def select_valid_window_indices(
    valid_indices: Sequence[int], selected_indices: Sequence[int]
) -> list[int]:
    selected = [int(index) for index in selected_indices]
    if len(selected) != len(set(selected)):
        raise ValueError("selected window indices must be unique")
    invalid = sorted(set(selected) - set(valid_indices))
    if invalid:
        raise ValueError(f"selected window indices are not valid starts: {invalid}")
    return selected
