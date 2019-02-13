# Please place imports here.
# BEGIN IMPORTS
import time
from math import floor
import numpy as np
import cv2
from scipy.sparse import csr_matrix
# import util_sweep
# END IMPORTS


def compute_photometric_stereo_impl(lights, images):
    """
    Given a set of images taken from the same viewpoint and a corresponding set
    of directions for light sources, this function computes the albedo and
    normal map of a Lambertian scene.

    If the computed albedo for a pixel has an L2 norm less than 1e-7, then set
    the albedo to black and set the normal to the 0 vector.

    Normals should be unit vectors.

    Input:
        lights -- N x 3 array.  Rows are normalized and are to be interpreted
                  as lighting directions.
        images -- list of N images.  Each image is of the same scene from the
                  same viewpoint, but under the lighting condition specified in
                  lights.
    Output:
        albedo -- float32 height x width x 3 image with dimensions matching the
                  input images.
        normals -- float32 height x width x 3 image with dimensions matching
                   the input images.
"""

    L = lights
    L_T = L.T
    albedo = np.zeros((images[0].shape[0], images[0].shape[1], images[0].shape[2]), dtype=np.float32)
    normals = np.zeros((images[0].shape[0], images[0].shape[1], 3), dtype=np.float32)
    term1 = np.linalg.inv(L_T.dot(L))
    for channel in range(images[0].shape[2]):
        for row in range(images[0].shape[0]):
            for col in range(images[0].shape[1]):
                I = [(images[i][row][col][channel]).T for i in range(len(images))]
                term2 = L_T.dot(I)  # LT*I
                G = term1.dot(term2)
                k = np.round(np.linalg.norm(G), 5)
                if k < 1e-7:
                    k = 0
                else:
                    normals[row][col] += G / k
                albedo[row][col][channel] = k
    normals /= images[0].shape[2]
    return albedo, normals



def project_impl(K, Rt, points):
    """
    Project 3D points into a calibrated camera.
    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        points -- height x width x 3 array of 3D points
    Output:
        projections -- height x width x 2 array of 2D projections
    """
    projection_matrix = K.dot(Rt)
    height, width = points.shape[:2]
    projections = np.zeros((height, width, 2))
    curr_point = np.zeros(3)

    for row_i, row in enumerate(points):
        for col_j, column in enumerate(row):
            curr_point = np.array(points[row_i, col_j])
            fourvec = np.array([curr_point[0], curr_point[1], curr_point[2], 1.0])
            homogenous_pt = projection_matrix.dot(fourvec)
            projections[row_i, col_j] = np.array(
                [homogenous_pt[0] / homogenous_pt[2], homogenous_pt[1] / homogenous_pt[2]])

    return projections



def preprocess_ncc_impl(image, ncc_size):
    """
    Prepare normalized patch vectors according to normalized cross
    correlation.

    This is a preprocessing step for the NCC pipeline.  It is expected that
    'preprocess_ncc' is called on every input image to preprocess the NCC
    vectors and then 'compute_ncc' is called to compute the dot product
    between these vectors in two images.

    NCC preprocessing has two steps.
    (1) Compute and subtract the mean.
    (2) Normalize the vector.

    The mean is per channel.  i.e. For an RGB image, over the ncc_size**2
    patch, compute the R, G, and B means separately.  The normalization
    is over all channels.  i.e. For an RGB image, after subtracting out the
    RGB mean, compute the norm over the entire (ncc_size**2 * channels)
    vector and divide.

    If the norm of the vector is < 1e-6, then set the entire vector for that
    patch to zero.

    Patches that extend past the boundary of the input image at all should be
    considered zero.  Their entire vector should be set to 0.

    Patches are to be flattened into vectors with the default numpy row
    major order.  For example, given the following
    2 (height) x 2 (width) x 2 (channels) patch, here is how the output
    vector should be arranged.

    channel1         channel2
    +------+------+  +------+------+ height
    | x111 | x121 |  | x112 | x122 |  |
    +------+------+  +------+------+  |
    | x211 | x221 |  | x212 | x222 |  |
    +------+------+  +------+------+  v
    width ------->

    v = [ x111, x121, x211, x221, x112, x122, x212, x222 ]

    see order argument in np.reshape

    Input:
        image -- height x width x channels image of type float32
        ncc_size -- integer width and height of NCC patch region.
    Output:
        normalized -- heigth x width x (channels * ncc_size**2) array
    """
    height, width, channels = image.shape
    window_offset = int(ncc_size / 2)
    normalized = np.zeros((height, width, (channels * (ncc_size ** 2))))#
    for row_i in range(window_offset, height - window_offset):
        for col_k in range(window_offset, width - window_offset):
            patch_vector = image[row_i - window_offset:row_i + window_offset + 1,
                           col_k - window_offset:col_k + window_offset + 1, :]
            mean_vec = np.mean(np.mean(patch_vector, axis=0), axis=0)#
            patch_vector = patch_vector - mean_vec

            temp_vec = np.zeros((channels * (ncc_size ** 2)))#

            big_index = 0

            for channel in range(channels):
                for row in range(patch_vector.shape[0]):
                    for col in range(patch_vector.shape[1]):
                        temp_vec[big_index] = patch_vector[row, col, channel]
                        big_index += 1

            patch_vector = temp_vec
            if (np.linalg.norm(patch_vector) >= 1e-6):#
                patch_vector /= np.linalg.norm(patch_vector)
            else:#
                patch_vector = np.zeros((channels * ncc_size ** 2))

            normalized[row_i, col_k] = patch_vector

    return normalized


def compute_ncc_impl(image1, image2):
    """

    Compute normalized cross correlation between two images that already have
    normalized vectors computed for each pixel with preprocess_ncc.

    Input:
        image1 -- height x width x (channels * ncc_size**2) array
        image2 -- height x width x (channels * ncc_size**2) array
    Output:
        ncc -- height x width normalized cross correlation between image1 and
               image2.
    """
    height, width = image1.shape[:2]
    ncc = np.zeros((height, width))

    for row_i in range(height):
        for col_k in range(width):
            ncc[row_i, col_k] = np.correlate(image1[row_i, col_k], image2[row_i, col_k])

    return ncc


def form_poisson_equation_impl(height, width, alpha, normals, depth_weight, depth):
    """
    Creates a Poisson equation given the normals and depth at every pixel in image.
    The solution to Poisson equation is the estimated depth.
    When the mode, is 'depth' in 'combine.py', the equation should return the actual depth.
    When it is 'normals', the equation should integrate the normals to estimate depth.
    When it is 'both', the equation should weight the contribution from normals and actual depth,
    using  parameter 'depth_weight'.

    Input:
        height -- height of input depth,normal array
        width -- width of input depth,normal array
        alpha -- stores alpha value of at each pixel of image.
            If alpha = 0, then the pixel normal/depth should not be
            taken into consideration for depth estimation
        normals -- stores the normals(nx,ny,nz) at each pixel of image
            None if mode is 'depth' in combine.py
        depth_weight -- parameter to tradeoff between normals and depth when estimation mode is 'both'
            High weight to normals mean low depth_weight.
            Giving high weightage to normals will result in smoother surface, but surface may be very different from
            what the input depthmap shows.
        depth -- stores the depth at each pixel of image
            None if mode is 'normals' in combine.py
    Output:
        constants for equation of type Ax = b
        A -- left-hand side coefficient of the Poisson equation
            note that A can be a very large but sparse matrix so csr_matrix is used to represent it.
        b -- right-hand side constant of the the Poisson equation
    """

    assert alpha.shape == (height, width)
    assert normals is None or normals.shape == (height, width, 3)
    assert depth is None or depth.shape == (height, width)

    '''
    Since A matrix is sparse, instead of filling matrix, we assign values to a non-zero elements only.
    For each non-zero element in matrix A, if A[i,j] = v, there should be some index k such that, 
        row_ind[k] = i
        col_ind[k] = j
        data_arr[k] = v
    Fill these values accordingly
    '''
    row_ind = []
    col_ind = []
    data_arr = []
    '''
    For each row in the system of equation fill the appropriate value for vector b in that row
    '''
    b = []
    if depth_weight is None:
        depth_weight = 1

    '''
    TODO
    Create a system of linear equation to estimate depth using normals and crude depth Ax = b

    x is a vector of depths at each pixel in the image and will have shape (height*width)
    A: ( k, height)
    x: ( height, width, 3)
    b: ( k, width)

    If mode is 'depth':
        > Each row in A and b corresponds to an equation at a single pixel
        > For each pixel k, 
            if pixel k has alpha value zero do not add any new equation.
            else, fill row in b with depth_weight*depth[k] and fill column k of the corresponding
                row in A with depth_weight.

        Justification: 
            Since all the elements except k in a row is zero, this reduces to 
                depth_weight*x[k] = depth_weight*depth[k]
            you may see that, solving this will give x with values exactly same as the depths, 
            at pixels where alpha is non-zero, then why do we need 'depth_weight' in A and b?
            The answer to this will become clear when this will be reused in 'both' mode

    Note: The normals in image are +ve when they are along an +x,+y,-z axes, if seen from camera's viewpoint.
    If mode is 'normals':
        > Each row in A and b corresponds to an equation of relationship between adjacent pixels
        > For each pixel k and its immideate neighbour along x-axis l
            if any of the pixel k or pixel l has alpha value zero do not add any new equation.
            else, fill row in b with nx[k] (nx is x-component of normal), fill column k of the corresponding
                row in A with -nz[k] and column k+1 with value nz[k]
        > Repeat the above along the y-axis as well, except nx[k] should be -ny[k].

        Justification: Assuming the depth to be smooth and almost planar within one pixel width.
        The normal projected in xz-plane at pixel k is perpendicular to tangent of surface in xz-plane.
        In other word if n = (nx,ny,-nz), its projection in xz-plane is (nx,nz) and if tangent t = (tx,0,tz),
            then n.t = 0, therefore nx/-nz = -tz/tx
        Therefore the depth change with change of one pixel width along x axis should be proporational to tz/tx = -nx/nz
        In other words (depth[k+1]-depth[k])*nz[k] = nx[k]
        This is exactly what the equation above represents.
        The negative sign in ny[k] is because the indexing along the y-axis is opposite of +y direction.

    If mode is 'both':
        > Do both of the above steps.

        Justification: The depth will provide a crude estimate of the actual depth. The normals do the smoothing of depth map
        This is why 'depth_weight' was used above in 'depth' mode. 
            If the 'depth_weight' is very large, we are going to give preference to input depth map.
            If the 'depth_weight' is close to zero, we are going to give preference normals.
    '''
    #TODO Block Begin
    #fill row_ind,col_ind,data_arr,b
    rn = 0
    for row_i in range(height):
        for col_j in range(width):
            k = row_i * width + col_j
            if alpha[row_i, col_j] != 0:
                if depth is not None:
                    b.append(depth_weight * depth[row_i, col_j])  # depth
                    row_ind.append(rn)  # depth
                    col_ind.append(k)  # depth
                    data_arr.append(depth_weight)  # depth
                    rn += 1

                if normals is not None:
                    if col_j + 1 <= width - 1 and alpha[row_i, col_j + 1] != 0:
                        # normals x-axis
                        b.append(normals[row_i, col_j, 0])
                        row_ind.append(rn)
                        col_ind.append(k)
                        data_arr.append(-normals[row_i, col_j, 2])
                        row_ind.append(rn)
                        col_ind.append(k + 1)
                        data_arr.append(normals[row_i, col_j, 2])
                        rn += 1
                    if row_i + 1 <= height - 1 and alpha[row_i + 1, col_j] != 0:
                        # normals mode y-axis
                        b.append(-normals[row_i, col_j, 1])
                        row_ind.append(rn)
                        col_ind.append(k)
                        data_arr.append(-normals[row_i, col_j, 2])
                        row_ind.append(rn)
                        col_ind.append(k + width)
                        data_arr.append(normals[row_i, col_j, 2])
                        rn += 1
    row = rn

    #TODO Block end
    # Convert all the lists to numpy array
    row_ind = np.array(row_ind, dtype=np.int32)
    col_ind = np.array(col_ind, dtype=np.int32)
    data_arr = np.array(data_arr, dtype=np.float32)
    b = np.array(b, dtype=np.float32)

    # Create a compressed sparse matrix from indices and values
    A = csr_matrix((data_arr, (row_ind, col_ind)), shape=(row, width * height))

    return A, b
