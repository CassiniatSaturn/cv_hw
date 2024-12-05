# Set up python path
import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

# Read data
img1 = cv2.imread("./data/NotreDame1.jpg")
color1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
gray1 = cv2.cvtColor(color1, cv2.COLOR_RGB2GRAY)

img2 = cv2.imread("./data/NotreDame2.jpg")
color2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
gray2 = cv2.cvtColor(color2, cv2.COLOR_RGB2GRAY)

print("Image 1 shape: ", gray1.shape)
print("Image 2 shape: ", gray2.shape)


# create SIFT, ORB
sift = cv2.SIFT_create()
orb = cv2.ORB_create()

# Detect keypoint use SIFT, ORB
kp1 = sift.detect(gray1, None)
img = cv2.drawKeypoints(gray1, kp1, img1)
kp2 = sift.detect(gray2, None)
img = cv2.drawKeypoints(gray2, kp2, img2)

# Detect keypoint using SIFT
kp1 = orb.detect(gray1, None)
img = cv2.drawKeypoints(gray1, kp1, img1)

# Detect keypoint using ORB
kp2 = orb.detect(gray2, None)
img = cv2.drawKeypoints(gray2, kp2, img2)

# Detect keypoint using Harris corner detection
dst1 = cv2.cornerHarris(gray1, 2, 3, 0.04)
harris_kp1 = np.stack((dst1 > 0.01 * dst1.max()).nonzero(), axis=1)
dst2 = cv2.cornerHarris(gray2, 2, 3, 0.04)
harris_kp2 = np.stack((dst2 > 0.01 * dst2.max()).nonzero(), axis=1)


def visualize_keypoint(keypoints, image):
    if len(image.shape) == 2:
        rgb_image = image.copy()
        rgb_image = rgb_image[..., None].repeat(3, -1)

    plt.imshow(rgb_image)
    fig = plt.gcf()
    ax = fig.gca()
    for kp in keypoints:
        ax.add_patch(plt.Circle((kp[1], kp[0]), 0.2, color="r"))

    plt.savefig("visualize_keypoints")
    plt.close()


# visualize_keypoint(harris_kp1, gray1)

# Compute the descriptor for extracted keypoint using SIFT
kp1, des1 = sift.compute(gray1, kp1)
kp2, des2 = sift.compute(gray2, kp2)


# Compute the descriptor for extracted keypoint using ORB
kp1, orb_des1 = orb.compute(gray1, kp1)
kp2, orb_des2 = orb.compute(gray2, kp2)


def cycle_consistency_match(descriptors1, descriptors2, norm="norm2"):
    """
    Brute-force descriptor match with Lowe tests and cross consistency check.


    Inputs:
    - descriptors1: tensor(N1, feature_size),the descriptors of a keypoint
    - descriptors2: tensor(N2, feature_size),the descriptors of a keypoint
    - device: where a torch.Tensor is or will be allocated
    - dist: distance metrics, hamming distance for measuring binary descriptor, and norm-2 distance for others
    - threshold: threshold for first Lowe test
    - ratio: ratio for second Lowe test

    Returns:
    - matches: tensor(M, 2), indices of corresponding matches in first and second set of descriptors,
      where matches[:, 0] denote the indices in the first and
      matches[:, 1] the indices in the second set of descriptors.
    """

    ######################################################################################################
    # TODO Q1: Find the indices of corresponding matches                                                 #
    # Use cross-consistency checking                               #
    ######################################################################################################

    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Compute distances
    # distances: [N1,N2]

    if norm == "hamming":
        # Convert to binary [N.256]
        binary_descriptors1 = np.unpackbits(descriptors1, axis=1)        
        binary_descriptors2 = np.unpackbits(descriptors2, axis=1)
        # Compute hamming distance [N1,N2]
        distances = np.count_nonzero(binary_descriptors1[:, None, :] != binary_descriptors2[None, :, :], axis=-1)

    else:
        distances = np.linalg.norm(
            descriptors1[:, None, :] - descriptors2[None, :, ...], axis=-1
        )

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
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return matches


def bf_match(descriptors1, descriptors2, norm="norm2", threshold=0, ratio=0.5):
    """
    Brute-force descriptor match with Lowe tests and cross consistency check.


    Inputs:
    - descriptors1: tensor(N, feature_size),the descriptors of a keypoint
    - descriptors2: tensor(N, feature_size),the descriptors of a keypoint
    - device: where a torch.Tensor is or will be allocated
    - dist: distance metrics, hamming distance for measuring binary descriptor, and norm-2 distance for others
    - threshold: threshold for first Lowe test
    - ratio: ratio for second Lowe test

    Returns:
    - matches: tensor(M, 2), indices of corresponding matches in first and second set of descriptors,
      where matches[:, 0] denote the indices in the first and
      matches[:, 1] the indices in the second set of descriptors.
    """

    ######################################################################################################
    # TODO Q1: Find the indices of corresponding matches                                                 #
    # See slide 48 of lecture 2 part A                                                                   #
    # Use Lowe test                                      #
    ######################################################################################################

    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Compute distances
    if norm == "hamming":
        # Convert to binary [N.256]
        binary_descriptors1 = np.unpackbits(descriptors1, axis=1)        
        binary_descriptors2 = np.unpackbits(descriptors2, axis=1)
        # Compute hamming distance [N1,N2]
        distances = np.count_nonzero(binary_descriptors1[:, None, :] != binary_descriptors2[None, :, :], axis=-1)

    else:
        distances = np.linalg.norm(
            descriptors1[:, None, :] - descriptors2[None, :, ...], axis=-1
        )
    # [N1,]
    indices1 = np.arange(descriptors1.shape[0])
    # top-2 matches [N1,2]
    indices2 = np.argsort(distances, axis=-1)[:, :2]

    # Lowe tests
    # dist_best: [N1,], dist_second_best:[N1,]
    dist_best = distances[indices1, indices2[:, 0]]
    dist_second_best = distances[indices1, indices2[:, 1]]
    dist_ratio = dist_best / dist_second_best
    mask = (dist_best < threshold) & (dist_ratio < ratio)
    indices1 = indices1[mask]
    indices2 = indices2[mask, 0]

    # Sort matches using distances (optional)
    dist = distances[indices1, indices2]
    sorted_indices = dist.argsort()

    matches = np.stack((indices1[sorted_indices], indices2[sorted_indices]), axis=1)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return matches


def match(descriptors1, descriptors2, device, dist="norm2", threshold=0, ratio=0.5):
    """
    Brute-force descriptor match with Lowe tests and cross consistency check.


    Inputs:
    - descriptors1: tensor(N, feature_size),the descriptors of a keypoint
    - descriptors2: tensor(N, feature_size),the descriptors of a keypoint
    - device: where a torch.Tensor is or will be allocated
    - dist: distance metrics, hamming distance for measuring binary descriptor, and norm-2 distance for others
    - threshold: threshold for first Lowe test
    - ratio: ratio for second Lowe test

    Returns:
    - matches: tensor(M, 2), indices of corresponding matches in first and second set of descriptors,
      where matches[:, 0] denote the indices in the first and
      matches[:, 1] the indices in the second set of descriptors.
    """

    # Exponent for norm
    if dist == "hamming":
        p = 0
    else:
        p = 2.0

    ######################################################################################################
    # TODO Q1: Find the indices of corresponding matches                                                 #
    # See slide 48 of lecture 2 part A                                                                   #
    # Use cross-consistency checking and first and second Lowe test                                      #
    ######################################################################################################

    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Compute distances
    distances = torch.cdist(
        descriptors1.squeeze().unsqueeze(0).to(torch.float32),
        descriptors2.squeeze().unsqueeze(0).to(torch.float32),
        p=p,
    ).squeeze()

    # Forward matching
    indices1 = torch.arange(descriptors1.shape[0]).to(device=device)
    _, indices2 = torch.topk(distances, k=2, dim=1, largest=False)

    # Lowe tests
    dist_best = distances[indices1, indices2[:, 0]]
    dist_second_best = distances[indices1, indices2[:, 1]]
    dist_ratio = torch.div(dist_best, dist_second_best)
    mask = torch.logical_and(dist_best < threshold, dist_ratio < ratio)
    indices1 = indices1[mask]
    indices2 = indices2[mask, 0]

    # Forward backward consistency check
    matches2 = torch.argmin(distances, dim=0)
    mask = (indices1 == matches2[indices2]).squeeze()
    indices1 = indices1[mask]
    indices2 = indices2[mask]

    # Sort matches using distances (optional)
    dist = distances[indices1, indices2]
    sorted_indices = dist.argsort()

    matches = torch.stack((indices1[sorted_indices], indices2[sorted_indices]), dim=1)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return matches


matches1 = cycle_consistency_match(descriptors1=orb_des1, descriptors2=orb_des2, norm="hamming")
print(matches1.shape)
matches2 = bf_match(
    descriptors1=orb_des1, descriptors2=orb_des2, threshold=170, ratio=0.5,norm="hamming"
)
print(matches2.shape)


def plot_matches(kp1, kp2, matches, image1, image2, title="matches"):
    new_shape1 = list(image1.shape)
    new_shape2 = list(image2.shape)

    if image1.shape[0] < image2.shape[0]:
        new_shape1[0] = image2.shape[0]
    elif image1.shape[0] > image2.shape[0]:
        new_shape2[0] = image1.shape[0]

    if image1.shape[1] < image2.shape[1]:
        new_shape1[1] = image2.shape[1]
    elif image1.shape[1] > image2.shape[1]:
        new_shape2[1] = image1.shape[1]

    if new_shape1 != image1.shape:
        new_image1 = np.zeros(new_shape1, dtype=image1.dtype)
        new_image1[: image1.shape[0], : image1.shape[1]] = image1
        image1 = new_image1

    if new_shape2 != image2.shape:
        new_image2 = np.zeros(new_shape2, dtype=image2.dtype)
        new_image2[: image2.shape[0], : image2.shape[1]] = image2
        image2 = new_image2

    offset = np.array(image1.shape)
    # align image horizontal
    image = np.concatenate([image1, image2], axis=1)
    offset[0] = 0

    fig, ax = plt.subplots()
    ax.imshow(image, cmap="gray")
    ax.axis((0, image1.shape[1] + offset[1], image2.shape[0] + offset[0], 0))

    rng = np.random.default_rng()

    for i in range(matches.shape[0]):
        idx0 = matches[i, 0]
        idx1 = matches[i, 1]

        color = rng.random(3)

        ax.plot(
            (kp1[idx0, 1], kp2[idx1, 1] + offset[1]),
            (kp1[idx0, 0], kp2[idx1, 0] + offset[0]),
            "-",
            color=color,
        )

    plt.savefig(f"./{title}.png")


def convert_kp_to_coord(keypoints):
    coord_list = []
    for kp in keypoints:
        coord_list.append(kp.pt)

    return np.array(coord_list)


plot_matches(
    convert_kp_to_coord(kp1),
    convert_kp_to_coord(kp2),
    matches1[:10, :],
    gray1,
    gray2,
    "cycle_consistency_match",
)
plot_matches(
    convert_kp_to_coord(kp1),
    convert_kp_to_coord(kp2),
    matches2[:10, :],
    gray1,
    gray2,
    "bf_match",
)


def cv2_matching():
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1.astype(np.uint8), des2.astype(np.uint8))

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw first 10 matches.
    img3 = cv2.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        matches[:10],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    plt.imshow(img3)
    plt.savefig("cv2_matching")
