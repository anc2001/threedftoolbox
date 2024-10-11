import json
import sys
import pymesh
from collections import defaultdict
import utils
import numpy as np
import math
from functools import reduce
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from scipy import ndimage
from PIL import Image, ImageDraw
import random
import trimesh
from shapely.geometry import Polygon
from threedftoolbox.render.render_depth_wall import render_mesh_vc
import threedftoolbox.scripts.utils as threedfutils
from threedftoolbox.scripts.scene import Instance, Furniture
import pickle

import argparse


def get_double_loop(vertices, loop, thickness=0.1, height=2.7):
    loop_vs = []
    loop_vs_outer = []
    thick_dirs = []
    z = vertices[loop[0][0]][1]
    for i, (uidx, vidx) in enumerate(loop):
        assert vertices[uidx][1] == z
        loop_vs.append(vertices[uidx][[0, 2]])
        # x0, z0, y0 = vertices[uidx]
        # x1, z1, y1 = vertices[vidx]
        # assert z0 == z1
    N = len(loop_vs)
    for j in range(N):
        i = (j - 1) % N
        k = (j + 1) % N

        a, o, b = loop_vs[i], loop_vs[j], loop_vs[k]

        thick_dir = np.asarray([o[1] - a[1], a[0] - o[0]])
        thick_dir = -thick_dir / np.linalg.norm(thick_dir)
        thick_dirs.append(thick_dir)

        oa = a - o
        ob = b - o
        oa = oa / np.linalg.norm(oa)
        ob = ob / np.linalg.norm(ob)

        mid = (oa + ob) / 2
        ao = -oa

        c = ao[0] * ob[1] - ao[1] * ob[0]
        # if np.dot(thick_dir, mid) > 0:
        if c == 0:
            mid = thick_dir * thickness
        else:
            mid = mid / np.linalg.norm(mid)
            coshalf = oa.dot(mid)
            sinhalf = (1 - coshalf**2) ** 0.5
            mid = mid * thickness / sinhalf
            if c < 0:
                mid = -mid
        loop_vs_outer.append(o + mid)

    loop_vs = np.asarray(loop_vs)
    loop_vs_outer = np.asarray(loop_vs_outer)
    return loop_vs, loop_vs_outer, thick_dirs


def perform_csg_operation(mesh1, mesh2, operation):
    mesh1 = pymesh.form_mesh(mesh1.vertices, mesh1.faces)
    mesh2 = pymesh.form_mesh(mesh2.vertices, mesh2.faces)

    csg_mesh = pymesh.boolean(mesh1, mesh2, operation=operation, engine="igl")

    # Convert PyMesh mesh back to Trimesh mesh
    result_mesh = trimesh.Trimesh(vertices=csg_mesh.vertices, faces=csg_mesh.faces)

    return result_mesh


def get_walls(
    loop_vs,
    loop_vs_outer,
    thick_dirs,
    doors,
    windows,
    thickness=0.1,
    wall_height=2.7,
    door_height=2.1,
    window_height=1.2,
    window_min_height=0.8,
):
    walls = []
    N = len(loop_vs)
    z = 0

    def create_cuboid(start, end, thick_dir, z, height):
        x0, y0 = start - thick_dir * (thickness * 0.1)
        x1, y1 = end - thick_dir * (thickness * 0.1)
        x01, y01 = start + thick_dir * thickness * 1.1
        x11, y11 = end + thick_dir * thickness * 1.1
        z1 = z + height

        cuboid_vertices = np.asarray(
            [
                [x0, z, y0],
                [x1, z, y1],
                [x01, z, y01],
                [x11, z, y11],
                [x0, z1, y0],
                [x1, z1, y1],
                [x01, z1, y01],
                [x11, z1, y11],
            ]
        )
        cuboid_faces = np.asarray(
            [
                [0, 1, 2],
                [2, 1, 3],
                [5, 4, 6],
                [5, 6, 7],
                [1, 0, 4],
                [1, 4, 5],
                [0, 2, 4],
                [4, 2, 6],
                [3, 1, 5],
                [3, 5, 7],
                [2, 3, 6],
                [6, 3, 7],
            ]
        )
        return trimesh.Trimesh(vertices=cuboid_vertices, faces=cuboid_faces)

    for i in range(N):
        j = (i + 1) % N

        x0, y0 = loop_vs[i]
        x1, y1 = loop_vs[j]

        x01, y01 = loop_vs_outer[i]
        x11, y11 = loop_vs_outer[j]

        z1 = z + wall_height

        wall_vertices = np.asarray(
            [
                [x0, z, y0],
                [x1, z, y1],
                [x01, z, y01],
                [x11, z, y11],
                [x0, z1, y0],
                [x1, z1, y1],
                [x01, z1, y01],
                [x11, z1, y11],
            ]
        )

        wall_faces = np.asarray(
            [
                [0, 1, 2],
                [2, 1, 3],
                [5, 4, 6],
                [5, 6, 7],
                [1, 0, 4],
                [1, 4, 5],
                [0, 2, 4],
                [4, 2, 6],
                [3, 1, 5],
                [3, 5, 7],
                [2, 3, 6],
                [6, 3, 7],
            ]
        )
        # wall_faces = np.asarray(
        #            [[0,1,2],[1,2,3],[4,5,6],[5,6,7],[0,1,4],[1,4,5],
        #        [0,2,4],[2,4,6],[1,3,5],[3,5,7],[2,3,6],[3,6,7]]
        # )
        thick_dir = thick_dirs[j]

        wall = trimesh.Trimesh(vertices=wall_vertices, faces=wall_faces)

        for wall_num, start, end in doors:
            if wall_num == i:
                door_cuboid = create_cuboid(start, end, thick_dir, z, door_height)
                wall = perform_csg_operation(wall, door_cuboid, "difference")

        # Subtract the windows from the walls
        for wall_num, start, end in windows:
            if wall_num == i:
                window_cuboid = create_cuboid(
                    start, end, thick_dir, z + window_min_height, window_height
                )
                wall = perform_csg_operation(wall, window_cuboid, "difference")

        # walls.append(wall)

        walls.append((wall.vertices, wall.faces))

    return walls


def create_floor_mesh(loop_vs_outer, floor_thickness=0.1):
    # Create shapely polygon for the top face
    top_face_polygon = Polygon(loop_vs_outer)

    if not top_face_polygon.is_valid:
        top_face_polygon = top_face_polygon.buffer(0)

    # Triangulate the polygon
    # print(loop_vs_outer)
    # print(top_face_polygon)
    top_face_vertices, top_face_faces = trimesh.creation.triangulate_polygon(
        top_face_polygon
    )

    # Add height component to top_face_vertices and bottom_face_vertices
    top_face_vertices = np.column_stack(
        (
            top_face_vertices[:, 0],
            np.zeros(top_face_vertices.shape[0]),
            top_face_vertices[:, 1],
        )
    )
    bottom_face_vertices = np.copy(top_face_vertices)
    bottom_face_vertices[:, 1] -= floor_thickness

    num_vertices = top_face_vertices.shape[0]
    vertices = np.vstack(
        (top_face_vertices, bottom_face_vertices)
    )  # + np.asarray([0,-0.005,0])
    faces = []

    # Connect top and bottom face vertices to create the side faces
    for i in range(num_vertices):
        j = (i + 1) % num_vertices
        faces.append([i, j + num_vertices, j])
        faces.append([i, i + num_vertices, j + num_vertices])

    # Add top_face_faces to faces, reversing the vertex order
    faces.extend(top_face_faces[:, ::-1])

    # Add bottom_face_faces to faces, adjusting indices
    bottom_face_faces = np.copy(top_face_faces) + num_vertices
    faces.extend(bottom_face_faces)

    # Create the mesh
    floor_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    return floor_mesh


def draw_walls_2D(
    loop_vs,
    loop_vs_outer,
    thick_dirs,
    doors,
    windows,
    img_size=512,
    bbox=(-3, 3, -3, 3),
    thickness=0.1,
):
    # Create a new image with a white background
    image = Image.new("RGB", (img_size, img_size), "white")
    draw = ImageDraw.Draw(image)

    xmin, xmax, ymin, ymax = bbox
    xy_max_size = max(xmax - xmin, ymax - ymin)
    xy_scale = img_size / xy_max_size

    # Calculate the translation factors
    loop_vs_min = loop_vs.min(axis=0)
    loop_vs_max = loop_vs.max(axis=0)
    xy_shift = (img_size / 2) - ((loop_vs_max + loop_vs_min) / 2) * xy_scale

    N = len(loop_vs)

    for i in range(N):
        j = (i + 1) % N

        v0 = loop_vs[i]
        v1 = loop_vs[j]
        v0_outer = loop_vs_outer[i]
        v1_outer = loop_vs_outer[j]

        # Scale and center the coordinates
        v0 = v0 * xy_scale + xy_shift
        v1 = v1 * xy_scale + xy_shift
        v0_outer = v0_outer * xy_scale + xy_shift
        v1_outer = v1_outer * xy_scale + xy_shift

        # Draw the wall as a black polygon with the specified thickness
        wall_points = [tuple(v0), tuple(v1), tuple(v1_outer), tuple(v0_outer)]
        draw.polygon(wall_points, fill="black")

        # Filter doors and windows on the current wall
        doors_on_wall = [door for door in doors if door[0] == i]
        windows_on_wall = [window for window in windows if window[0] == i]

        # wall_direction = v1 - v0
        # wall_perpendicular = np.array([-wall_direction[1], wall_direction[0]])
        # wall_perpendicular = wall_perpendicular / np.linalg.norm(wall_perpendicular)
        thick_dir = thick_dirs[j]  # note that it's j because I messed up

        # Draw doors as empty space
        for _, door_start, door_end in doors_on_wall:
            door_start = door_start * xy_scale + xy_shift
            door_end = door_end * xy_scale + xy_shift
            # door_offset = wall_perpendicular * (v0_outer - v0).dot(wall_perpendicular) / wall_perpendicular.dot(wall_perpendicular)
            door_offset = thick_dir * thickness * xy_scale
            door_points = [
                tuple(door_start),
                tuple(door_end),
                tuple(door_end + door_offset),
                tuple(door_start + door_offset),
            ]
            draw.polygon(door_points, fill="white")

        # Draw windows as thinner lines or hollow lines
        for _, window_start, window_end in windows_on_wall:
            window_start = window_start * xy_scale + xy_shift
            window_end = window_end * xy_scale + xy_shift
            window_offset = thick_dir * thickness * xy_scale
            # window_offset = wall_perpendicular * (v0_outer - v0).dot(wall_perpendicular) / wall_perpendicular.dot(wall_perpendicular)
            window_points = [
                tuple(window_start),
                tuple(window_end),
                tuple(window_end + window_offset),
                tuple(window_start + window_offset),
            ]

            # Erase a portion of the wall to avoid floating black lines
            draw.polygon(window_points, fill="white")

            # Redraw the window with a thinner line
            window_outline_points = [
                tuple(window_start),
                tuple(window_end),
                tuple(window_end + window_offset),
                tuple(window_start + window_offset),
            ]
            draw.polygon(window_outline_points, outline="black")

    # Return the image
    return image


def merge_repeated_vertices(vertices, faces):
    unique_vertices, unique_indices = np.unique(vertices, axis=0, return_inverse=True)
    # print(vertices)
    # print(unique_vertices[unique_indices])
    reindexed_faces = unique_indices[faces]
    return unique_vertices, reindexed_faces


def get_poly_outline(vertices, faces):
    edge_dict = {}
    for a, b, c in faces:
        edges = [(a, b), (a, c), (b, c)]
        for u, v in edges:
            if u > v:
                u, v = v, u

            if (u, v) not in edge_dict:
                edge_dict[(u, v)] = 1
            else:
                edge_dict[(u, v)] = edge_dict[(u, v)] + 1

    edge_candidates = [e for e in edge_dict.keys() if edge_dict[e] == 1]

    loop = [edge_candidates[0]]
    del edge_candidates[0]

    while len(edge_candidates) > 0:
        prev_end = loop[-1][1]
        removed = False
        for i, (u, v) in enumerate(edge_candidates):
            if u == prev_end:
                loop.append((u, v))
                del edge_candidates[i]
                removed = True
                break
            if v == prev_end:
                loop.append((v, u))
                del edge_candidates[i]
                removed = True
                break
        if not removed:
            return None

    return loop


def assert_clockwise(vertices, loop):
    s = 0
    for uidx, vidx in loop:
        x0, _, y0 = vertices[uidx]
        x1, _, y1 = vertices[vidx]
        s += (x1 - x0) * (y0 + y1)
    assert s > 0


def ensure_clockwise(vertices, loop):
    s = 0
    for uidx, vidx in loop:
        x0, _, y0 = vertices[uidx]
        x1, _, y1 = vertices[vidx]
        s += (x1 - x0) * (y0 + y1)
    if s > 0:  # clockwise
        return loop
    else:
        assert s < 0
        loop.reverse()
        loop = [(e[1], e[0]) for e in loop]
        assert_clockwise(vertices, loop)
        return loop


def simplify_poly_outline(vertices, loop):
    success = True
    while success:
        success = False
        for i in range(len(loop)):
            a, b = loop[i]
            x, c = loop[(i + 1) % len(loop)]
            assert b == x
            va, vb, vc = vertices[a], vertices[b], vertices[c]
            ab = vb - va
            bc = vc - vb
            angle = np.dot(ab, bc) / np.linalg.norm(ab) / np.linalg.norm(bc)
            if angle > 0.99:
                success = True
                old_loop = loop
                if i < len(loop) - 1:
                    loop = old_loop[:i] + [(a, c)] + old_loop[i + 2 :]
                else:
                    loop = old_loop[1:i] + [(a, c)]
                assert len(loop) + 1 == len(old_loop)
                break

    return loop


def ray_triangle_intersection(ray_origin, ray_direction, vertex0, vertex1, vertex2):
    EPSILON = 1e-8
    edge1 = vertex1 - vertex0
    edge2 = vertex2 - vertex0
    h = np.cross(ray_direction, edge2)
    a = np.dot(edge1, h)

    if -EPSILON < a < EPSILON:
        return None

    f = 1.0 / a
    s = ray_origin - vertex0
    u = f * np.dot(s, h)

    if u < 0.0 or u > 1.0:
        return None

    q = np.cross(s, edge1)
    v = f * np.dot(ray_direction, q)

    if v < 0.0 or u + v > 1.0:
        return None

    t = f * np.dot(edge2, q)

    if t > EPSILON:
        return t

    return None


def find_ray_intersections(x0, z0, x1, z1, x, y, z, verts, faces):
    ray_origin = np.array([x, y, z], dtype=np.float32)
    ray_direction = np.array([-(z1 - z0), 0.0, x1 - x0], dtype=np.float32)
    ray_direction /= np.linalg.norm(ray_direction)

    ray_origin = ray_origin - 10 * ray_direction

    intersections = []

    for face in faces:
        vertex0 = verts[face[0]]
        vertex1 = verts[face[1]]
        vertex2 = verts[face[2]]

        intersection = ray_triangle_intersection(
            ray_origin, ray_direction, vertex0, vertex1, vertex2
        )

        if intersection is not None:
            intersections.append(intersection)

    return intersections


def find_zero_regions(arr):
    # Set non-zero elements to 1
    arr[arr != 0] = 1

    # Label consecutive regions of zeros
    labeled_arr, num_features = ndimage.label(arr == 0)

    # Find the top-left and bottom-right corners of each region
    corners = []
    for feature in range(1, num_features + 1):
        coords = np.argwhere(labeled_arr == feature)
        top_left = coords.min(axis=0)
        bottom_right = coords.max(axis=0)
        corners.append((tuple(top_left), tuple(bottom_right)))

    return corners


def largest_rectangle_in_region(region):
    heights = np.zeros_like(region, dtype=int)
    max_area = 0
    max_rect = (0, 0, 0, 0)

    for i, row in enumerate(region):
        heights[i] = (heights[i - 1] + 1) * row if i > 0 else row

    for i, row in enumerate(heights):
        stack = []
        left_bound = np.zeros(len(row), dtype=int)
        right_bound = np.full(len(row), len(row), dtype=int)

        for j, h in enumerate(row):
            while stack and row[stack[-1]] >= h:
                right_bound[stack.pop()] = j
            left_bound[j] = stack[-1] if stack else -1
            stack.append(j)

        stack = []
        for j, h in reversed(list(enumerate(row))):
            while stack and row[stack[-1]] >= h:
                left_bound[stack.pop()] = j
            right_bound[j] = stack[-1] if stack else len(row)
            stack.append(j)

        for j, h in enumerate(row):
            area = (right_bound[j] - left_bound[j] - 1) * h
            if area > max_area:
                max_area = area
                max_rect = (i - h + 1, left_bound[j] + 1, i, right_bound[j] - 1)

    return max_rect


def find_largest_zero_rectangles(arr):
    arr[arr != 0] = 1
    labeled_arr, num_features = ndimage.label(arr == 0)

    rectangles = []
    for feature in range(1, num_features + 1):
        region = (labeled_arr == feature).astype(int)
        rect = largest_rectangle_in_region(region)
        if rect != (0, 0, 0, 0):
            rectangles.append(rect)

    return rectangles


def classify_region(
    zero_region,
    wall_start,
    wall_end,
    z_buffer_shape,
    door_min_aspect_ratio=1.5,
    door_max_aspect_ratio=3.5,
    window_min_aspect_ratio=0.5,
    window_max_aspect_ratio=3.0,
    door_min_width=0.5,
    window_min_width=0.35,
):
    y1, x1, y2, x2 = zero_region
    ymin, ymax = 1 - y2 / z_buffer_shape[0], 1 - y1 / z_buffer_shape[0]
    xmin, xmax = x1 / z_buffer_shape[1], x2 / z_buffer_shape[1]

    wall_start_2d = np.array([wall_start[0], wall_start[2]])
    wall_end_2d = np.array([wall_end[0], wall_end[2]])
    wall_length = np.linalg.norm(wall_end_2d - wall_start_2d)
    wall_dir = (wall_end_2d - wall_start_2d) / wall_length

    region_width = wall_length * (xmax - xmin)

    height_scale = wall_length / z_buffer_shape[1]
    region_height = height_scale * (y2 - y1)

    aspect_ratio = region_height / region_width

    touches_ceiling = ymax >= 0.95
    touches_ground = ymin <= 0.05

    start = wall_start_2d + wall_dir * wall_length * xmin
    end = wall_start_2d + wall_dir * wall_length * xmax

    # print(zero_region, touches_ground, touches_ceiling, aspect_ratio, region_width, region_height)
    if (
        touches_ground
        and door_min_aspect_ratio <= aspect_ratio <= door_max_aspect_ratio
        and region_width >= door_min_width
    ):
        return "door", start, end
    elif (
        not touches_ceiling
        and window_min_aspect_ratio <= aspect_ratio <= window_max_aspect_ratio
        and region_width >= window_min_width
    ):
        return "window", start, end
    else:
        return "neither", start, end


# ========================Object Parsing Code==================================
def z_angle(rotation):  # FRONT ATISS
    ref = [0, 0, 1]
    axis = np.cross(ref, rotation[1:])
    theta = np.arccos(np.dot(ref, rotation[1:])) * 2

    if np.sum(axis) == 0 or np.isnan(theta):
        return 0

    assert np.dot(axis, [1, 0, 1]) == 0
    assert 0 <= theta <= 2 * np.pi

    if theta >= np.pi:
        theta = theta - 2 * np.pi

    return np.sign(axis[1]) * theta


def draw_rectangle_transformed(
    vertices,
    instance,
    loop_vs,
    bbox=(-3, 3, -3, 3),
    border_width=4,
    img_size=512,
    color=(255, 255, 255),
):
    xmin, xmax, ymin, ymax = bbox
    xy_max_size = max(xmax - xmin, ymax - ymin)
    xy_scale = img_size / xy_max_size

    loop_vs_min = loop_vs.min(axis=0)
    loop_vs_max = loop_vs.max(axis=0)
    # xy_shift_room = (img_size / 2) - ((loop_vs_max + loop_vs_min) / 2) * xy_scale
    xshift_room, yshift_room = (loop_vs_min + loop_vs_max) / 2

    # Compute the size of the scaled rectangle
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    width = max_coords[0] - min_coords[0]
    height = max_coords[2] - min_coords[2]
    img_scale = img_size
    # Scale the rectangle
    width *= instance.scale[0] * xy_scale
    height *= instance.scale[2] * xy_scale

    # Create a new image to draw the rectangle
    img = Image.new("RGB", (int(width), int(height)), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rectangle(
        [0, 0, width, height], outline="black", width=border_width, fill=color
    )
    # Rotate the image
    # rotation_matrix = threedfutils.quaternion_to_matrix(instance.rot)
    # angle = np.arctan2(rotation_matrix[0, 1], rotation_matrix[0, 0]) * 180 / np.pi
    angle = z_angle(instance.rot) * 180 / np.pi % 360

    # Snap the angle to one of the four cardinal directions if it's within 5 degrees
    for cardinal_angle in [0, 90, 180, 270, 360]:
        if abs(angle - cardinal_angle) <= 5:
            angle = cardinal_angle
            break
    if angle == 360:
        angle == 0

    img = img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor="white")

    xshift = instance.pos[0] - xshift_room
    yshift = instance.pos[2] - yshift_room
    position = np.around(
        (xshift * xy_scale + img_size / 2, yshift * xy_scale + img_size / 2)
    ).astype(int)

    img_width, img_height = img.size
    top_left_x = position[0] - img_width // 2
    top_left_y = position[1] - img_height // 2
    paste_position = (top_left_x, top_left_y)

    # print("+++++++++++++++++++")
    # print(img_width, img_height, paste_position)
    # print(position, paste_position)
    # furniture_xy = [instance.pos[0] * xy_scale + xy_shift_room[0], instance.pos[2] * xy_scale + xy_shift_room[1]]
    # Create the final image and paste the rotated rectangle in the correct position
    final_img = Image.new("RGB", (img_size, img_size), (255, 255, 255))
    # position = np.round((furniture_xy[0] + min_coords[0], furniture_xy[1] + min_coords[2])).astype(int)
    final_img.paste(img, paste_position)
    return final_img


def blend_images(image1, image2):
    img1 = image1.convert("RGBA")
    img2 = image2.convert("RGBA")

    img1_pixels = img1.load()
    img2_pixels = img2.load()

    blended = Image.new("RGBA", img1.size)
    blended_pixels = blended.load()

    for y in range(img1.size[1]):
        for x in range(img1.size[0]):
            r1, g1, b1, a1 = img1_pixels[x, y]
            r2, g2, b2, a2 = img2_pixels[x, y]

            if (r1, g1, b1) == (255, 255, 255):  # White background
                blended_pixels[x, y] = img2_pixels[x, y]
            elif (r2, g2, b2) == (255, 255, 255):  # White background
                blended_pixels[x, y] = img1_pixels[x, y]
            elif (r1, g1, b1) == (0, 0, 0) or (r2, g2, b2) == (0, 0, 0):  # Black border
                blended_pixels[x, y] = (0, 0, 0, 255)
            else:  # Blend the colors
                r = (r1 + r2) // 2
                g = (g1 + g2) // 2
                b = (b1 + b2) // 2
                blended_pixels[x, y] = (r, g, b, 255)

    return blended.convert("RGB")


def save_obj_with_face_normals(
    vertices, faces, material_file_name, material_name, output_file
):
    # Compute face normals
    face_normals = np.cross(
        vertices[faces[:, 1]] - vertices[faces[:, 0]],
        vertices[faces[:, 2]] - vertices[faces[:, 0]],
    )
    face_normals /= np.linalg.norm(face_normals, axis=1)[:, np.newaxis]

    with open(output_file, "w") as f:
        f.write(f"mtllib {material_file_name}\n")

        # Write vertices
        for vertex in vertices:
            f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")

        # Write vertex normals (face normals)
        for normal in face_normals:
            f.write(f"vn {normal[0]} {normal[1]} {normal[2]}\n")

        # Write faces with vertex and normal indices
        f.write(f"usemtl {material_name}\n")
        for i, face in enumerate(faces):
            f.write(
                f"f {face[0] + 1}//{i + 1} {face[1] + 1}//{i + 1} {face[2] + 1}//{i + 1}\n"
            )


def combine_objs(objs):
    offset = 0
    verts = []
    faces = []
    for i, obj in enumerate(objs):
        v, f, _ = obj
        verts.append(v)
        faces.append(f + offset)
        offset += v.shape[0]

    verts = np.concatenate(verts, axis=0)
    faces = np.concatenate(faces, axis=0)
    return verts, faces


# =============================================================================


def parse_room(
    house_path,
    threedfuture_dir,
    output_path,
    room_type=None,
    save_obj=False,
    save_separate_obj=False,
    save_wall_vis=False,
    save_final=True,
    parse_wall=True,
):
    with open(house_path, "r") as f:
        house = json.load(f)

    house_name = house_path.stem

    furniture_in_scene = defaultdict()
    for furniture in house["furniture"]:
        if "valid" in furniture and furniture["valid"]:
            f = Furniture(
                furniture["uid"],
                furniture["jid"],
                model_info_dict[furniture["jid"]]["super-category"],
                model_info_dict[furniture["jid"]]["category"],
                furniture["size"] if "size" in furniture else None,
                furniture["bbox"] if "bbox" in furniture else None,
            )
            furniture_in_scene[furniture["uid"]] = f

    # Parse the extra meshes of the scene e.g walls, doors,
    # windows etc.
    meshes_in_scene = defaultdict()
    for mm in house["mesh"]:
        meshes_in_scene[mm["uid"]] = mm

    all_meshes = []
    for room in house["scene"]["room"]:
        room_id = room["instanceid"]
        if room_type is not None and room_type not in room_id.lower():
            continue
        room_dir = output_path / f"{house_name}_{room_id}"

        if (room_dir / "all_info.pkl").exists():
            continue

        room_debug_dir = room_dir / "debug"
        room_dir.mkdir(exist_ok=True)
        if save_separate_obj or save_obj: 
            room_debug_dir.mkdir(exist_ok=True)

        count = 0
        for child in room["children"]:
            ref = child["ref"]
            if ref in furniture_in_scene:
                pass
            elif ref in meshes_in_scene:
                mesh_data = meshes_in_scene[ref]
                verts = np.asarray(mesh_data["xyz"]).reshape(-1, 3)
                faces = np.asarray(mesh_data["faces"]).reshape(-1, 3)
                mesh_type = mesh_data["type"]

                all_meshes.append((verts, faces, mesh_type))
                if save_separate_obj:
                    utils.writeObj(
                        verts,
                        faces + 1,
                        room_debug_dir / f"{count}_{mesh_type}.obj",
                    )
                count += 1
            else:
                continue

        floor_meshes = [m for m in all_meshes if m[2] == "Floor"]
        if len(floor_meshes) == 0:
            continue
        floor_vs_original, floor_fs = merge_repeated_vertices(
            *combine_objs(floor_meshes)
        )
        if save_obj:
            utils.writeObj(
                floor_vs_original,
                floor_fs + 1,
                room_debug_dir / "all_floor.obj",
            )

        if not parse_wall:
            vertex_offset = np.zeros(3)
        else:
            outline = get_poly_outline(floor_vs_original, floor_fs)
            outline_original = ensure_clockwise(floor_vs_original, outline)
            outline_simplfied = simplify_poly_outline(floor_vs_original, outline_original)
            wall_vs_original, wall_fs = merge_repeated_vertices(
                *combine_objs(
                    [
                        m
                        for m in all_meshes
                        if not any(
                            a in m[2]
                            for a in ["Floor", "Door", "Pocket", "Ceiling", "Window"]
                        )
                    ]
                )
            )
            wall_vs_original = wall_vs_original.astype(float)
            if save_obj:
                utils.writeObj(
                    wall_vs_original,
                    wall_fs + 1,
                    room_debug_dir / "all_wall.obj",
                )

            y0 = wall_vs_original.min(axis=0)[1]
            y1 = wall_vs_original.max(axis=0)[1]
            sample_interval = 6 / 512  # corresponds to the floorplan vis

            for method, outline in [("", outline_simplfied)]:
                wall_imgs = []
                doors = []
                windows = []
                loop_vs, loop_vs_outer, thick_dirs = get_double_loop(
                    floor_vs_original, outline
                )

                loop_vs_min = loop_vs.min(axis=0)
                loop_vs_max = loop_vs.max(axis=0)
                loop_vs -= (loop_vs_min + loop_vs_max) / 2
                loop_vs_outer -= (loop_vs_min + loop_vs_max) / 2

                xshift_room, yshift_room = (loop_vs_min + loop_vs_max) / 2
                vertex_offset = np.asarray([xshift_room, 0, yshift_room])
                floor_vs = floor_vs_original - vertex_offset
                wall_vs = wall_vs_original - vertex_offset

                for wall_num, (u, v) in enumerate(outline):
                    x0, _, z0 = floor_vs[u]
                    x1, _, z1 = floor_vs[v]

                    front_dir = np.array((z0 - z1, 0, x1 - x0), dtype=np.float64)
                    front_dir /= np.linalg.norm(front_dir)
                    up_dir = np.array((0, 1, 0), dtype=np.float64)

                    right_dir = np.cross(front_dir, up_dir)
                    up_dir = np.cross(right_dir, front_dir)

                    transform = np.array(
                        (
                            (right_dir[0], front_dir[0], up_dir[0], 0),
                            (right_dir[1], front_dir[1], up_dir[1], 0),
                            (right_dir[2], front_dir[2], up_dir[2], 0),
                            (0, 0, 0, 1),
                        ),
                        dtype=np.float64,
                    )

                    bounds = np.array(
                        [
                            [x0, y0, z0, 1],
                            [x1, y1, z1, 1],
                        ],
                        dtype=np.float64,
                    )

                    bounds = np.dot(bounds, transform)
                    z_buffer = render_mesh_vc(
                        wall_vs,
                        wall_fs,
                        bbox=bounds,
                        img_size=None,
                        use_texture=False,
                        transform=transform,
                        scale=512 / 6,
                        wall_d_threshold=0.1,
                    )
                    z_buffer = np.abs(z_buffer)
                    z_buffer[z_buffer > 1] = 1
                    z_buffer = 1 - z_buffer
                    zero_regions = find_largest_zero_rectangles(np.copy(z_buffer))

                    z_buffer = z_buffer * 255
                    if save_wall_vis:
                        wall_rgb = np.stack((z_buffer, z_buffer, z_buffer), axis=2).astype(
                            "uint8"
                        )
                    for zero_region in zero_regions:
                        xx0, yy0, xx1, yy1 = zero_region
                        region_type, region_start, region_end = classify_region(
                            zero_region, floor_vs[u], floor_vs[v], z_buffer.shape
                        )
                        if region_type == "window":
                            if save_wall_vis:
                                wall_rgb[xx0:xx1, yy0:yy1, 2] = 128
                            windows.append((wall_num, region_start, region_end))
                        elif region_type == "door":
                            if save_wall_vis:
                                wall_rgb[xx0:xx1, yy0:yy1, 1] = 128
                            doors.append((wall_num, region_start, region_end))
                        else:
                            pass
                    if save_wall_vis:
                        wall_imgs.append(wall_rgb)

                if save_wall_vis:
                    line = np.zeros((wall_imgs[0].shape[0], 5, 3)).astype("uint8")
                    line[:, :, 0] = 255
                    vis = []
                    for img in wall_imgs:
                        vis.append(line)
                        vis.append(img)
                    vis.append(line)
                    images_combined = np.hstack(vis)
                    images_combined = Image.fromarray(images_combined)
                    images_combined.save(
                        room_debug_dir / f"walls{method}.png"
                    )

                walls = get_walls(loop_vs, loop_vs_outer, thick_dirs, doors, windows)
                floor_mesh = create_floor_mesh(loop_vs_outer)
                all_vs = [floor_mesh.vertices]
                all_fs = [floor_mesh.faces]
                offset = floor_mesh.vertices.shape[0]
                for i, (vs, fs) in enumerate(walls):
                    all_vs.append(vs)
                    fs = fs + offset
                    all_fs.append(fs)
                    offset += vs.shape[0]

                vroom = np.concatenate(all_vs, axis=0)
                froom = np.concatenate(all_fs, axis=0)
                if save_obj:
                    utils.writeObj(
                        vroom,
                        froom + 1,
                        room_debug_dir / f"procedural_wall{method}.obj",
                    )

        all_furnitures = []
        for child in room["children"]:
            ref = child["ref"]
            if ref in furniture_in_scene:
                finstance = Instance(
                    furniture_in_scene[ref],
                    child["pos"] - vertex_offset,
                    child["rot"],
                    child["scale"],
                )
                modelid = finstance.info.jid
                meshfile = str(threedfuture_dir / modelid  / "raw_model.obj")
                m = trimesh.load(meshfile, force='mesh')
                all_furnitures.append((m, finstance))

        all_transform = []
        for m, finstance in all_furnitures:
            modelid = finstance.info.jid

            rotation_m = np.array(threedfutils.quaternion_to_matrix(finstance.rot))
            affine_rot = np.zeros((4, 4))
            affine_rot[:3, :3] = np.transpose(
                rotation_m
            )  # not sure why but for now
            affine_rot[3, 3] = 1

            affine_scale = np.asarray(
                [
                    [finstance.scale[0], 0, 0, 0],
                    [0, finstance.scale[1], 0, 0],
                    [0, 0, finstance.scale[2], 0],
                    [0, 0, 0, 1],
                ]
            )

            affine_translation = np.asarray(
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [finstance.pos[0], finstance.pos[1], finstance.pos[2], 1],
                ]
            )

            affine = affine_rot @ affine_scale @ affine_translation
            all_transform.append(affine)

        do_intersections = True
        if do_intersections:
            transformed_meshes = []
            collisions = []

            for i in range(len(all_furnitures)):
                m = all_furnitures[i][0]
                transform = all_transform[i]
                v = m.vertices
                v = np.concatenate((v, np.ones((v.shape[0], 1))), axis=1)
                v = np.dot(v, transform)[:, :3]

                m2 = trimesh.Trimesh(vertices=v, faces=m.faces)

                m.vertices = v

                transformed_meshes.append(m)

            for i in range(len(transformed_meshes)):
                for j in range(i + 1, len(transformed_meshes)):
                    mesh1 = transformed_meshes[i]
                    mesh2 = transformed_meshes[j]

                    collision_manager = trimesh.collision.CollisionManager()
                    collision_manager.add_object("mesh1", mesh1)
                    collision_manager.add_object("mesh2", mesh2)
                    collision = collision_manager.in_collision_internal()

                    if collision:
                        collisions.append((i, j))
                        collisions.append((j, i))  # lazy
        else:
            collisions = []

        if save_obj:
            wall_mtl = "newmtl Wall \nKa 0.100000 0.100000 0.100000 \nKd 0.840000 0.840000 0.840000 \nKs 0.500000 0.500000 0.500000 \nNs 96.078431 \nNi 1.000000 \nd 1.000000 \nillum 1"
            with open(room_dir / "room.mtl", "w") as f:
                f.write(wall_mtl)

            save_obj_with_face_normals(
                vroom,
                froom,
                "room.mtl",
                "Wall",
                room_dir / "room.obj",
            )

        all_objs = []
        for i in range(len(all_furnitures)):
            m, finstance = all_furnitures[i]
            modelid = finstance.info.jid
            transform = all_transform[i]
            all_objs.append([modelid, transform])

        all_infos = {}
        all_infos["floor_verts"] = floor_vs_original
        all_infos["floor_fs"] = floor_fs 
        all_infos["vertex_offset"] = vertex_offset
        all_infos["all_objs"] = all_objs
        all_infos["furnitures"] = [fur[1] for fur in all_furnitures]
        all_infos["collisions"] = collisions
        all_infos["id"] = room_id
        if parse_wall:
            all_infos["loop_vs"] = loop_vs
            all_infos["loop_vs_outer"] = loop_vs_outer
            all_infos["thick_dirs"] = thick_dirs
            all_infos["doors"] = doors
            all_infos["windows"] = windows

        with open(room_dir / "all_info.pkl", "wb") as f:
            pickle.dump(all_infos, f, 2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--threedf-dir",
        type=Path,
        help="path to 3D-FRONT dataset",
    )
    parser.add_argument(
        "--threedfuture-dir",
        type=Path,
        help="path to 3D-FUTURE model directory",
    )
    parser.add_argument(
        "--model-info-path",
        type=Path,
        help="path to model_info.json",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        help="output path of parsing",
    )
    parser.add_argument(
        "--room-type",
        type=str,
        help="room type to filter",
        default=None,
    )
    parser.add_argument("--save-obj", action="store_true")
    parser.add_argument("--save-separate-obj", action="store_true")
    parser.add_argument("--save-wall-vis", action="store_true")
    parser.add_argument("--parse-wall", action="store_true")
    args = parser.parse_args()

    model_info = json.load(open(args.model_info_path, "r", encoding="utf-8"))
    model_info_dict = {}
    for model in model_info:
        model_info_dict[model["model_id"]] = model

    args.output_path.mkdir(exist_ok = True, parents = True)
    house_paths = list(args.threedf_dir.glob("*"))
    for house_path in tqdm(house_paths):
        parse_room(
            house_path,
            args.threedfuture_dir,
            args.output_path,
            room_type=args.room_type,
            save_obj=args.save_obj,
            save_separate_obj=args.save_separate_obj,
            save_wall_vis=args.save_wall_vis,
            parse_wall=args.parse_wall,
        )
