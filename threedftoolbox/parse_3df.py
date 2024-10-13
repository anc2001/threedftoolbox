import json
import sys
from collections import defaultdict
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

def merge_repeated_vertices(vertices, faces):
    unique_vertices, unique_indices = np.unique(vertices, axis=0, return_inverse=True)
    # print(vertices)
    # print(unique_vertices[unique_indices])
    reindexed_faces = unique_indices[faces]
    return unique_vertices, reindexed_faces

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

def parse_room(
    house_path,
    threedfuture_dir,
    output_path,
    room_type=None,
    do_intersections=True,
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
                count += 1
            else:
                continue

        floor_meshes = [m for m in all_meshes if m[2] == "Floor"]
        if len(floor_meshes) == 0:
            continue
        floor_vs_original, floor_fs = merge_repeated_vertices(
            *combine_objs(floor_meshes)
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

        all_objs = []
        for i in range(len(all_furnitures)):
            m, finstance = all_furnitures[i]
            modelid = finstance.info.jid
            transform = all_transform[i]
            all_objs.append([modelid, transform])

        all_infos = {}
        all_infos["floor_verts"] = floor_vs_original
        all_infos["floor_fs"] = floor_fs 
        all_infos["all_objs"] = all_objs
        all_infos["furnitures"] = [fur[1] for fur in all_furnitures]
        all_infos["collisions"] = collisions
        all_infos["id"] = room_id

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
        )
