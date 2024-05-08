import numpy as np

def normalize_rdm(RDM, exclude_diag=True):
    """
    Normalize a dissimilarity matrix (RDM) by calculating the z-score for each element.

    Parameters:
    - RDM (numpy.ndarray): The dissimilarity matrix to be normalized.
    - exclude_diag (bool): Whether to exclude the diagonal elements when calculating the z-score. 
                           Default is True.

    Returns:
    - numpy.ndarray: The normalized dissimilarity matrix.
    """

    if exclude_diag:
        # Create a mask to exclude the diagonal elements
        mask = ~np.eye(RDM.shape[0], dtype=bool)

        # Calculate mean and standard deviation excluding diagonal values
        mean_values = np.mean(RDM[mask])
        std_values = np.std(RDM[mask])

        # Calculate z-score for the entire matrix excluding diagonal values
        RDM = np.where(mask, (RDM - mean_values) / std_values, RDM)
        np.fill_diagonal(RDM, -10)

    else:
        # Calculate mean and standard deviation
        mean_values = np.mean(RDM)
        std_values = np.std(RDM)

        # Calculate z-score for the entire matrix
        RDM = (RDM - mean_values) / std_values

    return RDM
