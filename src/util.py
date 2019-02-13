import numpy as np
from math import floor
from itertools import izip
import cv2
import time
from student import compute_photometric_stereo_impl, \
    project_impl, \
    preprocess_ncc_impl, compute_ncc_impl,\
    form_poisson_equation_impl

from util_sweep import pyrup_impl, pyrdown_impl, unproject_corners_impl

def rerendering_error(lights, images, albedo, normals):
    errors = []
    for light, image in izip(lights.T, images):
        rerendered = albedo * \
            np.tensordot(normals, light.T, axes=1)[:, :, np.newaxis]
        error = (image - rerendered) / 255
        errors.append(np.sqrt((error ** 2).mean()))

    return sum(errors) / len(errors)


def form_poisson_equation(height, width, alpha, normals, depth_weight, depth):
    return form_poisson_equation_impl(height, width, alpha, normals, depth_weight, depth)


def compute_photometric_stereo(lights, images):
    return compute_photometric_stereo_impl(lights.T, images)


def project(K, Rt, points):
    return project_impl(K, Rt, points)


def unproject_corners(K, width, height, depth, Rt):
    return unproject_corners_impl(K, width, height, depth, Rt)


def flip_image(image):
    image = np.transpose(image, (1, 0, 2))
    image = image[::-1, :, :]
    return image


def compute_ncc(image1, image2):
    return compute_ncc_impl(image1, image2)


def preprocess_ncc(image, ncc_size):
    return preprocess_ncc_impl(image, ncc_size)


def get_depths(data):
    min_depth = data.min_depth
    max_depth = data.max_depth
    depth_layers = data.depth_layers
    min_depth_inv = 1.0 / min_depth
    max_depth_inv = 1.0 / max_depth
    step_inv = (min_depth_inv - max_depth_inv) / depth_layers
    depths = 1.0 / (np.arange(depth_layers) * step_inv + max_depth_inv)
    return np.float32(depths)


def pyrdown(image):
    return pyrdown_impl(image)


def pyrup(image):
    return pyrup_impl(image)


def save_mesh(K, width, height, albedo, normals, depth, alpha, filename):
    if albedo is not None:
        albedo = np.uint8(255.0 * albedo / albedo.max())
    else:
        albedo = 255 * np.ones((height, width, 3), dtype=np.uint8)

    if normals is None:
        normals = np.zeros((height, width, 3), dtype=np.float32)
        normals[:, :, 2] = -1
    else:
        normals = -normals

    if K is None:
        invK = None
    else:
        invK = np.linalg.inv(K)

    indices = np.nan * np.ones((height, width), dtype=np.float32)
    index = 0
    vertices = []
    for h in xrange(height):
        for w in xrange(width):
            if alpha[h, w]:
                indices[h, w] = index
                point = np.array(((w, ), (h, ), (1, )), dtype=np.float32)
                if invK is not None:
                    point = invK.dot(point)
                    point *= depth[h, w]
                else:
                    point[-1] *= depth[h, w]
                vertices.append('%f %f %f %f %f %f %d %d %d' % (
                    tuple(point.flatten().tolist()) +
                    tuple(normals[h, w, :].flatten().tolist()) +
                    tuple(albedo[h, w, :3].flatten().tolist())))
                index += 1

    faces = []
    for h in xrange(height - 1):
        for w in xrange(width - 1):
            index1 = indices[h, w]
            index2 = indices[h + 1, w]
            index3 = indices[h + 1, w + 1]
            index4 = indices[h, w + 1]
            corners = np.array((index1, index2, index3, index4))
            if not np.isnan(corners).any():
                faces.append('4 %d %d %d %d' %
                             (index1, index2, index3, index4))

    with open(filename, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('element vertex %d\n' % len(vertices))
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property float nx\n')
        f.write('property float ny\n')
        f.write('property float nz\n')
        f.write('property uchar diffuse_red\n')
        f.write('property uchar diffuse_green\n')
        f.write('property uchar diffuse_blue\n')
        f.write('element face %d\n' % len(faces))
        f.write('property list uint8 int32 vertex_index\n')
        f.write('end_header\n')

        f.write('\n'.join(vertices))
        f.write('\n')
        f.write('\n'.join(faces))


