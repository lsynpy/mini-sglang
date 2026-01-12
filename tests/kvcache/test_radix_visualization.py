import torch
from minisgl.kvcache.radix_manager import RadixCacheManager


def test_visualize_radix_tree():
    """Test the visualize_tree function of RadixCacheManager"""
    device = torch.device("cpu")
    manager = RadixCacheManager(device)

    # Create some test input sequences and indices
    input_ids1 = torch.tensor([101, 205, 307], dtype=torch.int32, device=device)
    indices1 = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)

    input_ids2 = torch.tensor([101, 205, 308], dtype=torch.int32, device=device)
    indices2 = torch.tensor([3, 4, 5], dtype=torch.int32, device=device)

    # Insert prefixes into the cache
    prefix_len1 = manager.insert_prefix(input_ids1, indices1)
    prefix_len2 = manager.insert_prefix(input_ids2, indices2)

    print(f"Inserted prefix 1 length: {prefix_len1}")
    print(f"Inserted prefix 2 length: {prefix_len2}")

    # Test visualization
    print("\nVisualizing radix tree:")
    manager.root_node.visualize()

    # Test matching prefixes
    match_handle1, match_indices1 = manager.match_prefix(input_ids1)
    print(
        f"\nMatched prefix 1: cached_len={match_handle1.cached_len}, indices shape={match_indices1.shape}"
    )

    match_handle2, match_indices2 = manager.match_prefix(input_ids2)
    print(
        f"Matched prefix 2: cached_len={match_handle2.cached_len}, indices shape={match_indices2.shape}"
    )

    # Test matching a partial prefix
    partial_input = torch.tensor([101, 205], dtype=torch.int32, device=device)
    match_handle3, match_indices3 = manager.match_prefix(partial_input)
    print(
        f"Matched partial prefix: cached_len={match_handle3.cached_len}, indices shape={match_indices3.shape}"
    )

    print("\nTest completed successfully!")


if __name__ == "__main__":
    test_visualize_radix_tree()
