import numpy as np


def compute_distance_matrix(descriptors1, descriptors2, norm="L2"):
    """
    Compute the distance matrix between two sets of descriptors.


    Inputs:
    - descriptors1: np.array(N1, feature_size),the descriptors of a keypoint
    - descriptors2: np.array(N2, feature_size),the descriptors of a keypoint

    Returns:
    - distances: np.array(N1, N2), the distance matrix between the two sets of descriptors.
    """

    assert norm in ["hamming", "L2"]

    if norm == "hamming":
        # Convert to binary [N.256]
        binary_descriptors1 = np.unpackbits(descriptors1, axis=1)
        binary_descriptors2 = np.unpackbits(descriptors2, axis=1)
        # Compute hamming distance [N1,N2]
        distances = np.count_nonzero(
            binary_descriptors1[:, None, :] != binary_descriptors2[None, :, :], axis=-1
        )

    else:
        distances = np.linalg.norm(
            descriptors1[:, None, :] - descriptors2[None, :, ...], axis=-1
        )

    return distances


def cycle_consistency_match(descriptors1, descriptors2, norm="L2"):
    """
    Brute-force descriptor match with cross consistency check.


    Inputs:
    - descriptors1: np.array(N1, feature_size),the descriptors of a keypoint
    - descriptors2: np.array(N2, feature_size),the descriptors of a keypoint

    Returns:
    - matches: np.array(M, 2), indices of corresponding matches in first and second set of descriptors,
      where matches[:, 0] denote the indices in the first and
      matches[:, 1] the indices in the second set of descriptors.
    """

    # Compute distances
    # distances: [N1,N2]
    distances = compute_distance_matrix(descriptors1, descriptors2, norm=norm)
    # indices1: [N1,]
    indice1 = np.arange(distances.shape[0])
    # Forward backward consistency check
    forward = np.argmin(distances, axis=-1)
    backward = np.argmin(distances[:, forward], axis=0)
    valid_mask = indice1 == backward

    matched_indice1 = indice1[valid_mask]
    macthed_indice2 = forward[valid_mask]

    # [M,2]
    dist = distances[matched_indice1, macthed_indice2]
    sorted_indices = dist.argsort()

    matches = np.stack(
        (matched_indice1[sorted_indices], macthed_indice2[sorted_indices]), axis=1
    )

    return matches


def lowe_match(descriptors1, descriptors2, threshold=0, ratio=0.5, norm="L2"):
    """
    Brute-force descriptor match with Lowe tests


    Inputs:
    - descriptors1: np.array(N, feature_size),the descriptors of a keypoint
    - descriptors2: np.array(N, feature_size),the descriptors of a keypoint
    - threshold: threshold for first Lowe test
    - ratio: ratio for second Lowe test

    Returns:
    - matches: np.array(M, 2), indices of corresponding matches in first and second set of descriptors,
      where matches[:, 0] denote the indices in the first and
      matches[:, 1] the indices in the second set of descriptors.
    """

    # Compute distances
    distances = compute_distance_matrix(descriptors1, descriptors2, norm=norm)
    # [N1,]
    indices1 = np.arange(descriptors1.shape[0])
    # top-2 matches [N1,2]
    indices2 = np.argsort(distances, axis=-1)[:, :2]

    # Lowe tests
    # dist_best: [N1,], dist_second_best:[N1,]
    dist_best = distances[indices1, indices2[:, 0]]
    print(dist_best.max())
    dist_second_best = distances[indices1, indices2[:, 1]]
    dist_ratio = dist_best / dist_second_best
    mask = (dist_best < threshold) & (dist_ratio < ratio)
    indices1 = indices1[mask]
    indices2 = indices2[mask, 0]

    # Sort matches using distances (optional)
    dist = distances[indices1, indices2]
    sorted_indices = dist.argsort()

    matches = np.stack((indices1[sorted_indices], indices2[sorted_indices]), axis=1)

    return matches
