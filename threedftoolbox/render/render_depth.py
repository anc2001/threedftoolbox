import trimesh
import numpy as np
import math
from numba import jit
from PIL import Image
import pickle
from tqdm import tqdm


def get_triangles(verts, faces):
    for face in faces:
        yield (
            list(verts[face[0]][:3]),
            list(verts[face[1]][:3]),
            list(verts[face[2]][:3]),
        )


def get_transform(xmin, xmax, ymin, ymax, zmin, zmax, img_size):
    assert ymax > ymin and xmax > xmin and zmax > zmin
    xy_max_size = max(xmax - xmin, ymax - ymin)
    xy_scale = img_size / xy_max_size
    z_scale = 1 / (zmax - xmin)

    t_scale = np.asarray(
        [[xy_scale, 0, 0, 0], [0, xy_scale, 0, 0], [0, 0, z_scale, 0], [0, 0, 0, 1]]
    )

    t_shift = np.asarray(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [-xmin, -ymin, -zmin, 1]]
    )

    return np.dot(t_shift, t_scale)


@jit(nopython=True)
def render(triangles, size, flat=False):
    result = np.zeros((size, size), dtype=np.float32)
    N, _, _ = triangles.shape

    for triangle in range(N):
        x0, y0, z0 = triangles[triangle][0]
        x1, y1, z1 = triangles[triangle][1]
        x2, y2, z2 = triangles[triangle][2]
        a = -y1 * x2 + y0 * (-x1 + x2) + x0 * (y1 - y2) + x1 * y2
        if a != 0:
            for i in range(
                max(0, math.floor(min(x0, x1, x2))),
                min(size, math.ceil(max(x0, x1, x2))),
            ):
                for j in range(
                    max(0, math.floor(min(y0, y1, y2))),
                    min(size, math.ceil(max(y0, y1, y2))),
                ):
                    x = i + 0.5
                    y = j + 0.5
                    s = (y0 * x2 - x0 * y2 + (y2 - y0) * x + (x0 - x2) * y) / a
                    t = (x0 * y1 - y0 * x1 + (y0 - y1) * x + (x1 - x0) * y) / a
                    if s < 0 and t < 0:
                        s = -s
                        t = -t
                    if 0 < s < 1 and 0 < t < 1 and s + t <= 1:
                        if flat:
                            result[i][j] = 1
                        else:
                            height = z0 * (1 - s - t) + z1 * s + z2 * t
                            result[i][j] = max(result[i][j], height)

    return result


def render_mesh(verts, faces, bbox=None, img_size=256, flat=False):
    # bbox is a list of xmin, xmax, ymin, ymax, zmin, zmax
    # if None, computed from mesh

    # verts = np.transpose(verts, (0,2,1))
    verts = verts[:, [2, 0, 1]]

    pads = np.ones((verts.shape[0], 1))
    verts = np.concatenate((verts, pads), axis=1)

    if bbox is None:
        xmin = verts[:, 0].min()
        xmax = verts[:, 0].max()
        ymin = verts[:, 1].min()
        ymax = verts[:, 1].max()
        zmin = verts[:, 2].min()
        zmax = verts[:, 2].max()

        xmin -= (xmax - xmin) * 0.1
        xmax += (xmax - xmin) * 0.1
        ymin -= (ymax - ymin) * 0.1
        ymax += (ymax - ymin) * 0.1
    else:
        xmin, xmax, ymin, ymax, zmin, zmax = bbox

    t = get_transform(xmin, xmax, ymin, ymax, zmin, zmax, img_size)

    verts = np.dot(verts, t)

    triangles = list(get_triangles(verts, faces))
    triangles = np.asarray(triangles, dtype=np.float32)
    img = render(triangles, img_size, flat)

    return img


if __name__ == "__main__":
    threedf_root_dir = "/data_hdd/3D-FRONT"
    threedfuture_dir = threedf_root_dir + "/3D-FUTURE-model"

    if __name__ == "__main__":
        # img = render_mesh('testobj.ply') * 255
        # img = Image.fromarray(img.astype('uint8'))
        # img.show()
        for i in tqdm(range(0, 1000)):
            try:
                bbox = [-3, 3, -3, 3, -0.5, 3]
                img_size = 512

                rootdir = f"render/3dfnew/{i}"
                m = trimesh.load(f"{rootdir}/room.obj")
                verts = [m.vertices]
                faces = [m.faces]
                offset = m.vertices.shape[0]

                with open(rootdir + "/test_scene.pkl", "rb") as f:
                    all_objs = pickle.load(f)

                for modelid, transform in all_objs:
                    meshfile = str(
                        threedfuture_dir + "/" + modelid + "/single_color_model.obj"
                    )
                    m = trimesh.load(meshfile)
                    v = m.vertices
                    v = np.concatenate((v, np.ones((v.shape[0], 1))), axis=1)
                    v = np.dot(v, transform)[:, :3]
                    verts.append(v)
                    faces.append(m.faces + offset)
                    offset += m.vertices.shape[0]

                verts = np.concatenate(verts, axis=0)
                faces = np.concatenate(faces, axis=0)

                # mesh = pmgr.create({
                #    'type': 'obj',
                #    'filename': rootdir+'/room.obj'
                # })
                # scene.addChild(mesh)

                # for modelid, transform in all_objs:

                img = render_mesh(verts, faces, bbox, img_size) * 255
                img = Image.fromarray(img.astype("uint8"))
                img.save(f"render/grammar_render/3dfdepth/{i}.png")
            except:
                print()
                print(f"{i}!!!!!")
        # img.save("test.png")
