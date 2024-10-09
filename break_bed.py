import os
import trimesh
import numpy as np
import itertools
from render_depth import render_mesh
import cv2
from collections import defaultdict


def combine_aabbs(aabbs):
    min_corner = np.array([min(aabb.bounds[0][i] for aabb in aabbs) for i in range(3)])
    max_corner = np.array([max(aabb.bounds[1][i] for aabb in aabbs) for i in range(3)])
    box = trimesh.creation.box(
        extents=max_corner - min_corner,
        transform=trimesh.transformations.translation_matrix(min_corner),
    )
    return box


def powerset(iterable):
    s = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(len(s) + 1)
    )


def split_obj_by_comment_test(input_obj_path):
    with open(input_obj_path, "r") as input_obj_file:
        lines = input_obj_file.readlines()

    current_object = None
    objects = {}
    vertices = []
    texture_coordinates = []

    for line in lines:
        if line.startswith("v "):
            vertices.append(line)
        elif line.startswith("vt "):
            texture_coordinates.append(line)
        elif line.startswith("# object"):
            current_object = line.strip().split(" ")[-1]
            if current_object not in objects:
                objects[current_object] = []
        elif line.startswith("f"):
            if current_object is not None:
                objects[current_object].append(line)
            else:
                if "single_object" not in objects:
                    objects["single_object"] = []
                objects["single_object"].append(line)
        elif current_object is not None:
            objects[current_object].append(line)

    meshes = []

    for obj, obj_lines in objects.items():
        obj_data = vertices + texture_coordinates + obj_lines
        mesh = trimesh.load_mesh(
            trimesh.util.wrap_as_stream("\n".join(obj_data)), file_type="obj"
        )
        mesh.name = obj
        meshes.append(mesh)

    return meshes


def split_obj_by_comment(input_obj_path):
    with open(input_obj_path, "r") as input_obj_file:
        lines = input_obj_file.readlines()

    current_group = "default_group"
    groups = defaultdict(list)
    vertices = []

    for line in lines:
        if line.startswith("v "):
            vertices.append(line)
        elif line.startswith("g "):
            current_group = line.strip().split(" ")[-1]
        elif line.startswith("f"):
            groups[current_group].append(line)

    meshes = []

    for group, group_lines in groups.items():
        group_data = vertices + group_lines
        mesh = trimesh.load_mesh(
            trimesh.util.wrap_as_stream("\n".join(group_data)), file_type="obj"
        )
        mesh.name = group
        meshes.append(mesh)

    return meshes


def count_non_empty_bins(normals, num_bins):
    bins = set()

    for normal in normals:
        try:
            phi, theta = normal_to_spherical(normal)
            phi_idx = int(phi / (2 * np.pi) * num_bins)
            theta_idx = int(theta / np.pi * num_bins)

            bin_idx = phi_idx * num_bins + theta_idx
        except:
            bin_idx = 0
        bins.add(bin_idx)

    return len(bins) / (num_bins * num_bins)


def calculate_normals(trimesh_obj):
    return trimesh_obj.vertex_normals


def normal_to_spherical(normal):
    x, y, z = normal
    r = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arctan2(y, x)
    theta = np.arccos(z / r)
    return phi, theta


def normal_coverage_score(component, num_bins=100):
    normals = component.face_normals
    score = count_non_empty_bins(normals, num_bins)

    return score


def solidness_score(component, num_bins=100):
    # For now, we only have one heuristic, but more can be added later.
    normal_score = normal_coverage_score(component, num_bins)
    # Inverting and squaring the score, so lower coverage results in a higher solidness score
    return (1.0 - normal_score) ** 2


def touches_ground_or_topmost(component, entire_mesh_aabb):
    component_aabb = component.bounding_box
    z_tolerance = 0.01 * entire_mesh_aabb.extents[1]

    touches_ground = (
        abs(component_aabb.bounds[0][1] - entire_mesh_aabb.bounds[0][1]) < z_tolerance
    )
    touches_topmost = (
        abs(component_aabb.bounds[1][1] - entire_mesh_aabb.bounds[1][1]) < z_tolerance
    )

    return touches_ground or touches_topmost


def find_best_solid_combination_old(components):
    aabb_entire_mesh = combine_aabbs(
        [component.bounding_box for component in components]
    )

    best_combo = []
    best_score = float("-inf")

    solidness_scores = [solidness_score(component) for component in components]
    is_ground_or_topmost = [
        touches_ground_or_topmost(component, aabb_entire_mesh)
        for component in components
    ]

    thresholds = [0, 0.6, 0.8, 0.9]
    solid_part_areas = []
    for threshold in thresholds:
        solid_parts = [
            components[idx].bounding_box
            for idx, solidness in enumerate(solidness_scores)
            if solidness > threshold
        ]
        if len(solid_parts) == 0:
            solid_part_areas.append(100000000000)
        else:
            solid_part_aabb = combine_aabbs(solid_parts)
            solid_part_areas.append(
                solid_part_aabb.extents[0] * solid_part_aabb.extents[2]
            )

    for combo in powerset(range(len(components))):
        if not combo:
            continue

        combined_aabb = combine_aabbs([components[i].bounding_box for i in combo])
        combined_area = combined_aabb.extents[0] * combined_aabb.extents[2]

        combined_solidness_score = sum(solidness_scores[i] for i in combo)
        combined_score = combined_solidness_score / len(combo)

        accepted = False
        for i, solid_part_area in enumerate(solid_part_areas):
            area_ratio = combined_area / solid_part_area
            area_threshold = 0.9 - 0.1 * i
            if area_ratio >= area_threshold:
                accepted = True
                break

        if not accepted:
            continue

        if combined_score > best_score:
            best_score = combined_score
            best_combo = combo

    return best_combo


def find_best_solid_combination(components):
    sorted_components = sorted(
        enumerate(components), key=lambda x: -solidness_score(x[1])
    )
    added_components_indices = []
    current_aabb = None

    for idx, component in sorted_components:
        component_aabb = component.bounding_box

        if current_aabb is None:
            current_aabb = component_aabb

        new_aabb = combine_aabbs([current_aabb, component_aabb])

        # Heuristic 1: Always add the component if it's solid enough
        if solidness_score(component) > 0.8:
            added_components_indices.append(idx)
            current_aabb = new_aabb
            continue

        if added_components_indices:
            # Compute the entire mesh AABB using the already added components
            entire_mesh_aabb = combine_aabbs(
                [components[i].bounding_box for i in added_components_indices]
            )

            # Check if the component touches the ground or is the topmost surface
            ground_or_topmost = touches_ground_or_topmost(component, entire_mesh_aabb)
        else:
            ground_or_topmost = False

        # Heuristic 2: Check if the existing parts already form the majority of the shape
        majority_shape = len(added_components_indices) / len(components) > 0.5

        # Heuristic 3: Adding the new part only extends the shape peripherally
        peripheral_extension = all(
            (new_aabb.extents[i] - current_aabb.extents[i]) / current_aabb.extents[i]
            < 0.1
            for i in (0, 2)
        )

        # If the component is not solid enough, and adding it would extend the shape peripherally,
        # check if the majority shape condition is met and if the component touches the ground or is the topmost surface
        if peripheral_extension and majority_shape and not ground_or_topmost:
            break

        added_components_indices.append(idx)
        current_aabb = new_aabb

    return added_components_indices


def save_meshes_to_obj(meshes, output_directory):
    os.makedirs(output_directory, exist_ok=True)

    for mesh in meshes:
        output_obj_path = os.path.join(output_directory, f"{mesh.name}.obj")
        with open(output_obj_path, "w") as output_obj_file:
            output_obj_file.write(trimesh.exchange.obj.export_obj(mesh))


if __name__ == "__main__":
    # bed_name = "1f7ba0ff-95b7-4fb1-8a39-412568de4b1b"
    threedf_root_dir = "/data_hdd/3D-FRONT"
    threedfuture_dir = f"{threedf_root_dir}/3D-FUTURE-model"

    # bed_dir = f"{threedfuture_dir}/{bed_name}"

    # input_obj = f"{bed_dir}/raw_model.obj"
    # output_dir = f"{bed_dir}/components"
    #
    # meshes = split_obj_by_comment(input_obj)
    # print(find_best_solid_combination(meshes))
    output_dir = "bed_debug"

    def process_beds(threedfuture_dir, render_dir):
        bed_names = [
            os.path.splitext(f)[0] for f in os.listdir(render_dir) if f.endswith(".png")
        ]

        for bed_name in bed_names:
            print(bed_name)
            bed_dir = f"{threedfuture_dir}/{bed_name}"
            input_obj = f"{bed_dir}/raw_model.obj"
            output_dir = "bed_debug"

            try:
                meshes = split_obj_by_comment(input_obj)

                # Render entire bed mesh
                aabb_entire_bed = combine_aabbs(
                    [component.bounding_box for component in meshes]
                )
                bed_extents = aabb_entire_bed.extents
                border_padding = 0.5
                bbox = [
                    -bed_extents[0] / 2 - border_padding,
                    bed_extents[0] / 2 + border_padding,
                    -bed_extents[2] / 2 - border_padding,
                    bed_extents[2] / 2 + border_padding,
                    -0.5,
                    bed_extents[1] + 0.5,
                ]

                bed_mesh = trimesh.util.concatenate(meshes)
                img = (
                    render_mesh(bed_mesh.vertices, bed_mesh.faces, bbox, img_size) * 255
                )
                cv2.imwrite(f"{output_dir}/{bed_name}_0.png", img)
                # Render each individual component
                # for i, component in enumerate(meshes):
                #    img = render_mesh(component.vertices, component.faces, bbox, img_size) * 255
                #    cv2.imwrite(f"{output_dir}/{bed_name}_{i+1}.png", img)

                # Render the best solid combination
                best_combo = find_best_solid_combination(meshes)
                print(best_combo)
                combined_mesh = trimesh.util.concatenate(
                    [meshes[i] for i in best_combo]
                )
                # combined_mesh = trimesh.util.concatenate(best_combo)
                img = (
                    render_mesh(
                        combined_mesh.vertices, combined_mesh.faces, bbox, img_size
                    )
                    * 255
                )
                cv2.imwrite(f"{output_dir}/{bed_name}_100.png", img)
            except Exception as e:
                if isinstance(e, KeyboardInterrupt):
                    print("Interrupted by user. Exiting...")
                    quit()
                else:
                    raise
                    continue
                    # print(f"Caught an exception: {e}. Continuing execution...")

    threedf_root_dir = "/data_hdd/3D-FRONT"
    threedfuture_dir = f"{threedf_root_dir}/3D-FUTURE-model"
    render_dir = "render/bed_renders"
    img_size = 512
    bbox = [-3, 3, -3, 3, -0.5, 3]

    process_beds(threedfuture_dir, render_dir)
