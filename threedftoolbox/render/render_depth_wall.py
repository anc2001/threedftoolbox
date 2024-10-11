import trimesh
import numpy as np
import math
from numba import jit
from PIL import Image


def get_triangles(verts, faces):
    for face in faces:
        yield (
            list(verts[face[0]][:3]),
            list(verts[face[1]][:3]),
            list(verts[face[2]][:3]),
        )


def get_transform(xmin, xmax, ymin, ymax, zmin, zmax, img_size):
    assert ymax > ymin and xmax > xmin and zmax > zmin
    xz_max_size = max(xmax - xmin, zmax - zmin)
    xz_scale = img_size / xz_max_size
    y_scale = 1 / (ymax - ymin)

    t_scale = np.asarray(
        [[xz_scale, 0, 0, 0], [0, y_scale, 0, 0], [0, 0, xz_scale, 0], [0, 0, 0, 1]]
    )

    t_shift = np.asarray(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [-xmin, -ymin, -zmin, 1]]
    )

    return np.dot(t_shift, t_scale)


def get_transform_fixed_scale(xmin, xmax, ymin, ymax, zmin, zmax, scale):
    assert ymax >= ymin and xmax >= xmin and zmax >= zmin
    xz_max_size = max(xmax - xmin, zmax - zmin)
    xz_scale = scale
    y_scale = 1

    t_scale = np.asarray(
        [[xz_scale, 0, 0, 0], [0, y_scale, 0, 0], [0, 0, xz_scale, 0], [0, 0, 0, 1]]
    )

    t_shift = np.asarray(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [-xmin, -(ymin + ymax) / 2, -zmin, 1],
        ]
    )

    return np.dot(t_shift, t_scale)


# @jit(nopython=True)
# def render(triangles, size):
#    y_buffer = np.zeros((size, size), dtype=np.float32)
#    t_index = np.zeros((size, size), dtype=np.int64) - 1
#
#    N, _, _ = triangles.shape
#
#    for triangle in range(N):
#        x0,y0,z0 = triangles[triangle][0]
#        x1,y1,z1 = triangles[triangle][1]
#        x2,y2,z2 = triangles[triangle][2]
#        a = -y1*x2 + y0*(-x1+x2) + x0*(y1-y2) + x1*y2
#        if a != 0:
#            for i in range(max(0,math.floor(min(x0,x1,x2))), \
#                            min(size,math.ceil(max(x0,x1,x2)))):
#                for j in range(max(0,math.floor(min(y0,y1,y2))), \
#                                min(size,math.ceil(max(y0,y1,y2)))):
#                    x = i+0.5
#                    y = j+0.5
#                    s = (y0*x2 - x0*y2 + (y2-y0)*x + (x0-x2)*y)/a
#                    t = (x0*y1 - y0*x1 + (y0-y1)*x + (x1-x0)*y)/a
#                    if s < 0 and t < 0:
#                        s = -s
#                        t = -t
#                    if 0 < s < 1 and 0 < t < 1 and s + t <= 1:
#                        height = z0 *(1-s-t) + z1*s + z2*t
#                        if height > y_buffer[i][j]:
#                            y_buffer[i][j] = height
#                            t_index[i][j] = triangle
#
#    return y_buffer, t_index


@jit(nopython=True)
def render(verts, faces, fts, vts, midxs, mat_imgs, size, use_texture=True):
    xsize, ysize = size
    y_buffer = np.zeros((xsize, ysize), dtype=np.float64) - 1000
    rendered = np.zeros((xsize, ysize, 3), dtype=np.float64)

    N = faces.shape[0]
    # print("ashfauihfaishfiasi")
    # print(faces.shape, fts.shape)
    # print("ashfauihfaishfiasi")
    # print("ashfauihfaishfiasi")
    # print("ashfauihfaishfiasi")
    for triangle in range(N):
        x0, y0, z0 = verts[faces[triangle][0]]
        x1, y1, z1 = verts[faces[triangle][1]]
        x2, y2, z2 = verts[faces[triangle][2]]
        if use_texture:
            tx0, ty0 = vts[fts[triangle][0]]
            tx1, ty1 = vts[fts[triangle][1]]
            tx2, ty2 = vts[fts[triangle][2]]
        # print(tx0, ty0)
        a = -z1 * x2 + z0 * (-x1 + x2) + x0 * (z1 - z2) + x1 * z2
        if a != 0:
            for i in range(
                max(0, math.floor(min(x0, x1, x2))),
                min(ysize, math.ceil(max(x0, x1, x2))),
            ):
                for j in range(
                    max(0, math.floor(min(z0, z1, z2))),
                    min(xsize, math.ceil(max(z0, z1, z2))),
                ):
                    x = i + 0.5
                    z = j + 0.5
                    s = (z0 * x2 - x0 * z2 + (z2 - z0) * x + (x0 - x2) * z) / a
                    t = (x0 * z1 - z0 * x1 + (z0 - z1) * x + (x1 - x0) * z) / a
                    if s < 0 and t < 0:
                        s = -s
                        t = -t
                    else:
                        pass
                    if 0 < s < 1 and 0 < t < 1 and s + t <= 1:
                        height = y0 * (1 - s - t) + y1 * s + y2 * t
                        cur_depth = y_buffer[xsize - 1 - j][ysize - 1 - i]
                        if (
                            (height >= 0 and cur_depth <= 0)
                            or (height >= 0 and height < cur_depth)
                            or (height <= 0 and height > cur_depth)
                        ):
                            # if height > y_buffer[xsize-1-j][ysize-1-i]:
                            y_buffer[xsize - 1 - j][ysize - 1 - i] = height

                            # tx, ty = int((tx)*2048+0.5), int((1-ty)*2048+0.5)
                            if use_texture:
                                tx = tx0 * (1 - s - t) + tx1 * s + tx2 * t
                                ty = ty0 * (1 - s - t) + ty1 * s + ty2 * t
                                tx, ty = (tx) * 2047, (1 - ty) * 2047
                                tx, ty = int(round(tx)), int(round(ty))
                                assert 0 <= tx < 2048
                                assert 0 <= ty < 2048
                                mat_img = mat_imgs[midxs[triangle]]
                                rendered[xsize - 1 - j][ysize - 1 - i] = mat_img[ty][tx]
                            else:
                                rendered[xsize - 1 - j][ysize - 1 - i] = 255

    return y_buffer, rendered


# @jit(nopython=True)
def render_vc(verts, faces, size):
    xsize, ysize = size
    y_buffer = np.zeros((xsize, ysize), dtype=np.float64) - 1000
    # rendered = np.zeros((xsize, ysize, 3), dtype=np.float64)

    N = faces.shape[0]
    # print("ashfauihfaishfiasi")
    # print(faces.shape, fts.shape)
    # print("ashfauihfaishfiasi")
    # print("ashfauihfaishfiasi")
    # print("ashfauihfaishfiasi")
    for triangle in range(N):
        x0, y0, z0 = verts[faces[triangle][0]]
        x1, y1, z1 = verts[faces[triangle][1]]
        x2, y2, z2 = verts[faces[triangle][2]]

        a = -z1 * x2 + z0 * (-x1 + x2) + x0 * (z1 - z2) + x1 * z2
        if a != 0:
            for i in range(
                max(0, math.floor(min(x0, x1, x2))),
                min(ysize, math.ceil(max(x0, x1, x2))),
            ):
                for j in range(
                    max(0, math.floor(min(z0, z1, z2))),
                    min(xsize, math.ceil(max(z0, z1, z2))),
                ):
                    x = i + 0.5
                    z = j + 0.5
                    s = (z0 * x2 - x0 * z2 + (z2 - z0) * x + (x0 - x2) * z) / a
                    t = (x0 * z1 - z0 * x1 + (z0 - z1) * x + (x1 - x0) * z) / a
                    if s < 0 and t < 0:
                        s = -s
                        t = -t
                    else:
                        pass
                    if 0 < s < 1 and 0 < t < 1 and s + t <= 1:
                        height = y0 * (1 - s - t) + y1 * s + y2 * t
                        cur_depth = y_buffer[xsize - 1 - j][ysize - 1 - i]
                        if (
                            (height >= 0 and cur_depth <= 0)
                            or (height >= 0 and height < cur_depth)
                            or (height <= 0 and height > cur_depth)
                        ):
                            # if height > y_buffer[xsize-1-j][ysize-1-i]:
                            y_buffer[xsize - 1 - j][ysize - 1 - i] = height
                            # result[i][j] = max(result[i][j], height)
                            # tx, ty = int((tx)*2048+0.5), int((1-ty)*2048+0.5)
                            # color = c0 * (1-s-t) + c1 * s + c2*t
                            # color = color[:3]
                            # rendered[xsize-1-j][ysize-1-i] = color

    return y_buffer


def render_mesh(
    verts,
    faces,
    fts,
    vts,
    midxs,
    mat_imgs,
    bbox=None,
    img_size=256,
    use_texture=True,
    transform=None,
    scale=None,
    wall_d_threshold=None,
):
    # bbox is a list of xmin, xmax, ymin, ymax, zmin, zmax
    # if None, computed from mesh

    # m = trimesh.load('testobj.ply')

    # verts = m.vertices
    # faces = m.faces
    pads = np.ones((verts.shape[0], 1))
    verts = np.concatenate((verts, pads), axis=1)

    if transform is not None:
        verts = np.dot(verts, transform)

    if img_size is None:
        assert (
            scale is not None
            and bbox is not None
            and transform is not None
            and wall_d_threshold is not None
        )
        xmin = bbox[:, 0].min()
        xmax = bbox[:, 0].max()
        ymin = bbox[:, 1].min() - wall_d_threshold
        ymax = bbox[:, 1].max() + wall_d_threshold
        zmin = bbox[:, 2].min()
        zmax = bbox[:, 2].max()

        ysize = math.ceil((xmax - xmin) * scale)
        xsize = math.ceil((zmax - zmin) * scale)

        t = get_transform_fixed_scale(xmin, xmax, ymin, ymax, zmin, zmax, scale)
    else:
        raise NotImplementedError

    # print(f"Image size {xsize} x {ysize}")

    verts = np.dot(verts, t)
    bbox = np.dot(bbox, t)
    # print("=====Bounds After Transform=====")
    # print(bbox)
    # print(verts.min(axis=0))
    # print(verts.max(axis=0))
    # print('================================')

    verts = verts[:, :3]

    # print(verts.shape)
    # verts = np.transpose(verts, (1,2,0))
    # verts= verts[:,[0,2,1]]

    img = render(
        verts, faces, fts, vts, midxs, mat_imgs, (xsize, ysize), use_texture=use_texture
    )

    return img


def render_mesh_vc(
    verts,
    faces,
    bbox=None,
    img_size=256,
    use_texture=True,
    transform=None,
    scale=None,
    wall_d_threshold=None,
):
    # bbox is a list of xmin, xmax, ymin, ymax, zmin, zmax
    # if None, computed from mesh

    # m = trimesh.load('testobj.ply')

    # verts = m.vertices
    # faces = m.faces
    pads = np.ones((verts.shape[0], 1))
    verts = np.concatenate((verts, pads), axis=1)

    if transform is not None:
        verts = np.dot(verts, transform)

    if img_size is None:
        assert (
            scale is not None
            and bbox is not None
            and transform is not None
            and wall_d_threshold is not None
        )
        xmin = bbox[:, 0].min()
        xmax = bbox[:, 0].max()
        ymin = bbox[:, 1].min() - wall_d_threshold
        ymax = bbox[:, 1].max() + wall_d_threshold
        zmin = bbox[:, 2].min()
        zmax = bbox[:, 2].max()

        ysize = math.ceil((xmax - xmin) * scale)
        xsize = math.ceil((zmax - zmin) * scale)

        t = get_transform_fixed_scale(xmin, xmax, ymin, ymax, zmin, zmax, scale)
    else:
        raise NotImplementedError

    # print(f"Image size {xsize} x {ysize}")

    verts = np.dot(verts, t)
    bbox = np.dot(bbox, t)
    # print("=====Bounds After Transform=====")
    # print(bbox)
    # print(verts.min(axis=0))
    # print(verts.max(axis=0))
    # print('================================')

    verts = verts[:, :3]

    # print(verts.shape)
    # verts = np.transpose(verts, (1,2,0))
    # verts= verts[:,[0,2,1]]

    # img = render(verts, faces, fts, vts, midxs, mat_imgs, (xsize, ysize), use_texture=use_texture)
    img = render_vc(verts, faces, (256, ysize))

    return img


if __name__ == "__main__":
    # img = render_mesh('testobj.ply') * 255
    # img = Image.fromarray(img.astype('uint8'))
    # img.show()
    bbox = [-40, 360, -120, 280, 0, 170]
    img_size = 1024
    img = render_mesh("testobj.ply", bbox, img_size) * 255
    img = Image.fromarray(img.astype("uint8"))
    img.show()
    # img.save("test.png")
