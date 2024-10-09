import gzip
import math
import os
import os.path
import sys
import pickle
from contextlib import contextmanager
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np


def show_img(array):
    try:
        img = Image.fromarray(array.astype("uint8"))
    except:
        img = array
    img.show()


def show_img_color(array):
    try:
        img = Image.fromarray(np.rollaxis(np.rollaxis(array, 2), 2).astype("uint8"))
    except:
        img = array
    img.show()


def save_img(array, dirr):
    img = Image.fromarray(array.astype("uint8"))
    img.save(dirr)


def save_img_color(array, dirr):
    img = Image.fromarray(np.rollaxis(np.rollaxis(array, 2), 2).astype("uint8"))
    img.save(dirr)


"""
Find the first item in a list that satisfies a predicate
"""


def find(pred, lst):
    for item in lst:
        if pred(item):
            return item


"""
Ensure a directory exists
"""


def ensuredir(dirname):
    """
    Ensure a directory exists
    """
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def pickle_dump_compressed(object, filename, protocol=pickle.HIGHEST_PROTOCOL):
    """
    Pickles + compresses an object to file
    """
    file = gzip.GzipFile(filename, "wb")
    file.write(pickle.dumps(object, protocol))
    file.close()


def pickle_load_compressed(filename):
    """
    Loads a compressed pickle file and returns reconstituted object
    """
    file = gzip.GzipFile(filename, "rb")
    buffer = b""
    while True:
        data = file.read()
        if data == b"":
            break
        buffer += data
    object = pickle.loads(buffer)
    file.close()
    return object


def get_data_root_dir():
    """
    Gets the root dir of the dataset
    Check env variable first,
    if not set, use the {code_location}/data
    """
    env_path = os.environ.get("SCENESYNTH_DATA_PATH")
    if env_path:
        # if False: #Debug purposes
        return env_path
    else:
        root_dir = os.path.dirname(os.path.abspath(__file__))
        return "/data_hdd/3D-FRONT"


@contextmanager
def stdout_redirected(to=os.devnull):
    """
    From https://stackoverflow.com/questions/5081657/how-do-i-prevent-a-c-shared-library-to-print-on-stdout-in-python
    Suppress C warnings
    """
    fd = sys.stdout.fileno()

    ##### assert that Python and C stdio write using the same file descriptor
    ####assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_stdout(to):
        sys.stdout.close()  # + implicit flush()
        os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, "w")  # Python writes to fd

    with os.fdopen(os.dup(fd), "w") as old_stdout:
        with open(to, "w") as file:
            _redirect_stdout(to=file)
        try:
            yield  # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout)  # restore stdout.
            # buffering and flags such as
            # CLOEXEC may be different


def sample_surface(faces, vs, count, return_normals=True):
    """
    sample mesh surface
    sample method:
    http://mathworld.wolfram.com/TrianglePointPicking.html
    Args
    ---------
    vs: vertices (batch x nvs x 3d coordinate)
    faces: triangle faces (torch.long) (num_faces x 3)
    count: number of samples
    Return
    ---------
    samples: (count, 3) points in space on the surface of mesh
    face_index: (count,) indices of faces for each sampled point
    """
    if torch.isnan(faces).any() or torch.isnan(vs).any():
        assert False, "saw nan in sample_surface"

    device = vs.device
    bsize, nvs, _ = vs.shape
    area, normal = face_areas_normals(faces, vs)
    area_sum = torch.sum(area, dim=1)

    assert not (area <= 0.0).any().item(), "Saw negative probability while sampling"
    assert not (area_sum <= 0.0).any().item(), "Saw negative probability while sampling"
    assert not (area > 1000000.0).any().item(), "Saw inf"
    assert not (area_sum > 1000000.0).any().item(), "Saw inf"

    dist = torch.distributions.categorical.Categorical(probs=area / (area_sum[:, None]))
    face_index = dist.sample((count,))

    # pull triangles into the form of an origin + 2 vectors
    tri_origins = vs[:, faces[:, 0], :]
    tri_vectors = vs[:, faces[:, 1:], :].clone()
    tri_vectors -= tri_origins.repeat(1, 1, 2).reshape((bsize, len(faces), 2, 3))

    # pull the vectors for the faces we are going to sample from
    face_index = face_index.transpose(0, 1)
    face_index = face_index[:, :, None].expand((bsize, count, 3))
    tri_origins = torch.gather(tri_origins, dim=1, index=face_index)
    face_index2 = face_index[:, :, None, :].expand((bsize, count, 2, 3))
    tri_vectors = torch.gather(tri_vectors, dim=1, index=face_index2)

    # randomly generate two 0-1 scalar components to multiply edge vectors by
    random_lengths = torch.rand(count, 2, 1, device=vs.device, dtype=tri_vectors.dtype)

    # points will be distributed on a quadrilateral if we use 2x [0-1] samples
    # if the two scalar components sum less than 1.0 the point will be
    # inside the triangle, so we find vectors longer than 1.0 and
    # transform them to be inside the triangle
    random_test = random_lengths.sum(dim=1).reshape(-1) > 1.0
    random_lengths[random_test] -= 1.0
    random_lengths = torch.abs(random_lengths)

    # multiply triangle edge vectors by the random lengths and sum
    sample_vector = (tri_vectors * random_lengths[None, :]).sum(dim=2)

    # finally, offset by the origin to generate
    # (n,3) points in space on the triangle
    samples = sample_vector + tri_origins

    if return_normals:
        samples = torch.cat(
            (samples, torch.gather(normal, dim=1, index=face_index)), dim=2
        )
        return samples
    else:
        return samples


def face_areas_normals(faces, vs):
    face_normals = torch.cross(
        vs[:, faces[:, 1], :] - vs[:, faces[:, 0], :],
        vs[:, faces[:, 2], :] - vs[:, faces[:, 1], :],
        dim=2,
    )
    face_areas = torch.norm(face_normals, dim=2) + 1e-8
    face_normals = face_normals / face_areas[:, :, None]
    face_areas = 0.5 * face_areas
    return face_areas, face_normals


def load_obj(fn):
    fin = open(fn, "r")
    lines = [line.rstrip() for line in fin]
    fin.close()

    vertices = []
    faces = []
    for line in lines:
        if line.startswith("v "):
            vertices.append(np.float32(line.split()[1:4]))
        elif line.startswith("f "):
            faces.append(np.int32([item.split("/")[0] for item in line.split()[1:4]]))

    f = np.vstack(faces)
    v = np.vstack(vertices)
    return v, f


def writeObj(verts, faces, outfile, mtl=None):
    # faces = faces.clone()
    # faces += 1
    with open(outfile, "w") as f:
        if mtl is not None:
            f.write(f"{mtl[0]}\n")

        for a, b, c in verts.tolist():
            f.write(f"v {a} {b} {c}\n")

        if mtl is not None:
            f.write(f"{mtl[1]}\n")
        for a, b, c in faces.tolist():
            f.write(f"f {a} {b} {c}\n")
