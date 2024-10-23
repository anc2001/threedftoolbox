import json
import yaml
import pickle
import argparse
from collections import defaultdict
import numpy as np
import math
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import random
import trimesh

from shapely.geometry import Polygon
import threedftoolbox.scripts.utils as threedfutils
from threedftoolbox.scripts.scene import Instance, Furniture
from scipy.spatial.transform import Rotation


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
    unique_scene_ids,
    house_path,
    threedfuture_dir,
    output_path,
    room_type=None,
    do_intersections=True,
    invalid_scene_ids=[],
    invalid_jids=[],
    largest_allowed_dim=None,
    skip=False,
):
    with open(house_path, "r") as f:
        house = json.load(f)

    house_name = house_path.stem

    furniture_in_scene = defaultdict()
    for furniture in house["furniture"]:
        if (
            "valid" in furniture
            and furniture["valid"]
            and furniture["jid"] not in invalid_jids
        ):
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

    for room in house["scene"]["room"]:
        room_id = room["instanceid"]

        if room_type is not None and room_type not in room_id.lower():
            continue

        if room_id in unique_scene_ids:
            continue

        room_dir = output_path / f"{house_name}_{room_id}"
        if skip and (room_dir / "all_info.pkl").exists():
            unique_scene_ids.append(room_id)
            continue

        # Furniture in room
        furniture_in_room = []
        # Extra meshes in room
        extra_meshes_in_room = []
        for child in room["children"]:
            ref = child["ref"]
            if child["ref"] in furniture_in_scene:
                # If scale is very small/big ignore this scene
                if any(si < 1e-5 for si in child["scale"]):
                    return
                if any(si > 5 for si in child["scale"]):
                    return

                finstance = Instance(
                    furniture_in_scene[ref],
                    child["pos"],
                    child["rot"],
                    child["scale"],
                )
                modelid = finstance.info.jid
                meshfile = str(threedfuture_dir / modelid / "raw_model.obj")
                m = trimesh.load(meshfile, force="mesh")
                furniture_in_room.append((m, finstance))
            elif child["ref"] in meshes_in_scene:
                mesh_data = meshes_in_scene[ref]
                verts = np.asarray(mesh_data["xyz"]).reshape(-1, 3)
                faces = np.asarray(mesh_data["faces"]).reshape(-1, 3)
                mesh_type = mesh_data["type"]

                extra_meshes_in_room.append((verts, faces, mesh_type))
            else:
                continue

        floor_meshes = [m for m in extra_meshes_in_room if m[2] == "Floor"]
        if len(floor_meshes) == 0:
            continue
        floor_vs_original, floor_fs = merge_repeated_vertices(
            *combine_objs(floor_meshes)
        )
        floor_min_bound = np.amin(floor_vs_original, axis=0)
        floor_max_bound = np.amax(floor_vs_original, axis=0)
        floor_centroid = np.mean([floor_min_bound, floor_max_bound], axis=0)

        bboxes = []
        for m, finstance in furniture_in_room:
            object_verts = np.array(m.vertices)
            object_verts *= finstance.scale
            object_min_bound = np.amin(object_verts, axis=0)
            object_max_bound = np.amax(object_verts, axis=0)
            extent = (object_max_bound - object_min_bound) / 2

            rotvec = Rotation.from_quat(finstance.rot).as_rotvec()
            theta = rotvec[1]
            R = np.zeros((3, 3))
            R[0, 0] = np.cos(theta)
            R[0, 2] = -np.sin(theta)
            R[2, 0] = np.sin(theta)
            R[2, 2] = np.cos(theta)
            R[1, 1] = 1
            object_verts = object_verts.dot(R) + finstance.pos - floor_centroid
            object_min_bound = np.amin(object_verts, axis=0)
            object_max_bound = np.amax(object_verts, axis=0)
            center = np.mean([object_min_bound, object_max_bound], axis=0)

            bbox = {"rotation": rotvec[1:2], "size": extent, "translation": center}
            bboxes.append(bbox)

        all_infos = {}
        all_infos["floor_verts"] = floor_vs_original - floor_centroid
        if largest_allowed_dim:
            scene_min_bound = np.amin(all_infos["floor_verts"], axis=0)
            scene_max_bound = np.amax(all_infos["floor_verts"], axis=0)
            scene_extent = scene_max_bound - scene_min_bound
            if (
                scene_extent[0] > largest_allowed_dim
                or scene_extent[2] > largest_allowed_dim
            ):
                return

        all_infos["floor_fs"] = floor_fs
        if len(furniture_in_room) == 0:
            return

        all_infos["furnitures"] = [fur[1] for fur in furniture_in_room]
        all_infos["bboxes"] = bboxes
        all_infos["scene_id"] = room_id

        room_dir.mkdir(exist_ok=True)
        with open(room_dir / "all_info.pkl", "wb") as f:
            pickle.dump(all_infos, f, 2)

        unique_scene_ids.append(room_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--threedf-dir",
        type=Path,
        required=True,
        help="path to 3D-FRONT dataset",
    )
    parser.add_argument(
        "--threedfuture-dir",
        type=Path,
        required=True,
        help="path to 3D-FUTURE model directory",
    )
    parser.add_argument(
        "--model-info-path",
        type=Path,
        required=True,
        help="path to model_info.json",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="output path of parsing",
    )
    parser.add_argument(
        "--bounds-file",
        type=Path,
        help="option to specify largest allowed scene size, room-type must be specified",
        default=None,
    )
    parser.add_argument(
        "--room-type",
        type=str,
        required=True,
        help="room type to filter",
        default=None,
    )
    parser.add_argument(
        "--invalid-jids",
        type=str,
        help="text file of black list of invalid houses",
        default=None,
    )
    parser.add_argument(
        "--invalid-scene-ids",
        type=str,
        help="text file of invalid rooms",
        default=None,
    )
    parser.add_argument(
        "--skip",
        action="store_true",
        help="whether to skip rooms that have already been parsed",
    )
    args = parser.parse_args()

    model_info = json.load(open(args.model_info_path, "r", encoding="utf-8"))
    model_info_dict = {}
    for model in model_info:
        model_info_dict[model["model_id"]] = model

    if args.invalid_jids:
        with open(args.invalid_jids, "rb") as f:
            invalid_jids = set(l.strip() for l in f)
    else:
        invalid_jids = []

    if args.invalid_scene_ids:
        with open(args.invalid_scene_ids, "rb") as f:
            invalid_scene_ids = set(l.strip() for l in f)
    else:
        invalid_scene_ids = []

    if args.bounds_file:
        with open(args.bounds_file, "rb") as f:
            bounds_config = yaml.safe_load(f)
        if not args.room_type:
            raise ValueError(
                f"room type should be specified to specify largest allowed scene size"
            )
        largest_allowed_dim = bounds_config[args.room_type]["largest_allowed_dim"]
    else:
        largest_allowed_dim = None

    args.output_path.mkdir(exist_ok=True, parents=True)
    house_paths = list(args.threedf_dir.glob("*"))
    unique_scene_ids = []
    for house_path in tqdm(house_paths):
        parse_room(
            unique_scene_ids,
            house_path,
            args.threedfuture_dir,
            args.output_path,
            room_type=args.room_type,
            invalid_scene_ids=invalid_scene_ids,
            invalid_jids=invalid_jids,
            largest_allowed_dim=largest_allowed_dim,
            skip=args.skip,
        )
