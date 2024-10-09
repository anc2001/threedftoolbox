from PIL import Image, ImageOps, ImageDraw, ImageChops
from pathlib import Path
from tqdm import tqdm
import pickle
import trimesh
from numba import jit
import numpy as np
import math
import random
import json
import pymesh
from threedftoolbox.scripts.scene import Instance, Furniture
from render_depth import render_mesh
import break_bed
import cv2
from skimage.measure import label, regionprops
from shapely.geometry import box, Polygon
from shapely.ops import unary_union
from itertools import cycle
import os


def draw_double_bed(img, color, border_color="black", border_width=4):
    min_coord = (0, 0)
    max_coord = img.size
    width = max_coord[0] - min_coord[0]
    height = max_coord[1] - min_coord[1]

    draw = ImageDraw.Draw(img)

    # Draw the bedframe
    draw.rectangle(
        [min_coord, max_coord], outline=border_color, width=border_width, fill=color
    )

    # Draw the mattress
    spacing = border_width * 2
    mattress_width = width - spacing * 2
    mattress_height = int(height * 0.75) - spacing * 2

    mattress_x = min_coord[0] + spacing
    mattress_y = min_coord[1] + int(height * 0.25) + spacing

    draw.rectangle(
        [
            (mattress_x, mattress_y),
            (mattress_x + mattress_width, mattress_y + mattress_height),
        ],
        outline=border_color,
        width=border_width,
    )

    pillow_width = int(width * 0.44)
    pillow_height = int(height * 0.25) - spacing * 2
    pillow_space = int(width * 0.05)

    left_pillow_x = min_coord[0] + pillow_space
    right_pillow_x = min_coord[0] + width - pillow_space - pillow_width
    left_pillow_y = min_coord[1] + spacing
    right_pillow_y = min_coord[1] + spacing

    draw.rectangle(
        [
            (left_pillow_x, left_pillow_y),
            (left_pillow_x + pillow_width, left_pillow_y + pillow_height),
        ],
        outline=border_color,
        width=border_width,
    )
    draw.rectangle(
        [
            (right_pillow_x, right_pillow_y),
            (right_pillow_x + pillow_width, right_pillow_y + pillow_height),
        ],
        outline=border_color,
        width=border_width,
    )

    return img


def draw_single_bed(img, color, border_color="black", border_width=4):
    min_coord = (0, 0)
    max_coord = img.size
    width = max_coord[0] - min_coord[0]
    height = max_coord[1] - min_coord[1]

    draw = ImageDraw.Draw(img)

    # Draw the bedframe
    draw.rectangle(
        [min_coord, max_coord], outline=border_color, width=border_width, fill=color
    )

    # Draw the mattress
    spacing = border_width * 2
    mattress_width = width - spacing * 2
    mattress_height = int(height * 0.75) - spacing * 2

    mattress_x = min_coord[0] + spacing
    mattress_y = min_coord[1] + int(height * 0.25) + spacing

    draw.rectangle(
        [
            (mattress_x, mattress_y),
            (mattress_x + mattress_width, mattress_y + mattress_height),
        ],
        outline=border_color,
        width=border_width,
    )

    pillow_width = mattress_width
    pillow_height = int(height * 0.25) - spacing * 2

    pillow_x = min_coord[0] + spacing
    pillow_y = min_coord[1] + spacing

    draw.rectangle(
        [(pillow_x, pillow_y), (pillow_x + pillow_width, pillow_y + pillow_height)],
        outline=border_color,
        width=border_width,
    )

    return img


def draw_chair(img, color, border_color="black", border_width=4):
    min_coord = (0, 0)
    max_coord = img.size
    width = max_coord[0] - min_coord[0]
    height = max_coord[1] - min_coord[1]

    draw = ImageDraw.Draw(img)

    # Draw the chair back
    back_width = int(width * 0.7) + border_width
    back_height = int(height * 0.15)
    back_x = min_coord[0] + int(width * 0.15) - border_width / 2
    back_y = min_coord[1]

    draw.rectangle(
        [(back_x, back_y), (back_x + back_width, back_y + back_height)],
        outline=border_color,
        width=2,
        fill=color,
    )

    # Draw the chair seat
    seat_width = width * 0.7 + border_width
    seat_x = width * 0.15 - border_width / 2
    seat_height = int(height * 0.85) + border_width / 2
    seat_y = min_coord[1] + back_height - border_width / 2

    # Draw the chair arms
    arm_height = seat_height * 0.9
    arm_y = seat_y + seat_height * 0.05
    arm_width = int(width * 0.15)
    arm_x_left = min_coord[0]
    arm_x_right = max_coord[0] - arm_width - 1

    draw.rectangle(
        [(arm_x_left, arm_y), (arm_x_left + arm_width, arm_y + arm_height)],
        outline=border_color,
        width=2,
        fill=color,
    )
    draw.rectangle(
        [(arm_x_right, arm_y), (arm_x_right + arm_width, arm_y + arm_height)],
        outline=border_color,
        width=2,
        fill=color,
    )

    draw.rectangle(
        [(seat_x, seat_y), (seat_x + seat_width, seat_y + seat_height)],
        outline=border_color,
        width=border_width,
        fill=color,
    )

    return img


def draw_sofa(img, color, border_color="black", border_width=4):
    min_coord = (0, 0)
    max_coord = img.size
    width = max_coord[0] - min_coord[0]
    height = max_coord[1] - min_coord[1]

    draw = ImageDraw.Draw(img)

    # Draw the sofa base
    draw.rectangle(
        [min_coord, max_coord], outline=border_color, width=border_width, fill=color
    )

    # Draw the sofa back
    back_height = int(height * 0.2)
    draw.rectangle(
        [(min_coord[0], min_coord[1]), (max_coord[0], min_coord[1] + back_height)],
        outline=border_color,
        width=border_width,
        fill=color,
    )

    # Draw the armrests
    arm_width = int(width * 0.15)
    arm_height = int(height * 0.8) + border_width // 2
    arm_y = min_coord[1] + back_height - border_width // 2

    draw.rectangle(
        [(min_coord[0], arm_y), (min_coord[0] + arm_width, arm_y + arm_height)],
        outline=border_color,
        width=border_width,
        fill=color,
    )
    draw.rectangle(
        [(max_coord[0] - arm_width, arm_y), (max_coord[0], arm_y + arm_height)],
        outline=border_color,
        width=border_width,
        fill=color,
    )

    # Draw cushions only if width > height
    if width > height:
        # Calculate the number of cushions based on width-to-height ratio
        cushion_width = (width - 2 * arm_width) / int((width - 2 * arm_width) / height)
        num_cushions = int((width - 2 * arm_width) / cushion_width)

        # Draw the cushions
        for i in range(num_cushions):
            cushion_x = min_coord[0] + arm_width + i * cushion_width
            cushion_y = arm_y
            draw.rectangle(
                [
                    (cushion_x, cushion_y),
                    (cushion_x + cushion_width, cushion_y + arm_height),
                ],
                outline=border_color,
                width=border_width,
            )

    return img


def draw_rectangle(img, color, border_color="black", border_width=4):
    min_coord = (0, 0)
    max_coord = img.size
    width = max_coord[0] - min_coord[0]
    height = max_coord[1] - min_coord[1]

    draw = ImageDraw.Draw(img)

    draw.rectangle(
        [min_coord, max_coord], outline=border_color, width=border_width, fill=color
    )

    return img


def draw_polygon(img, coords, color, border_color="black", border_width=4):
    draw = ImageDraw.Draw(img)

    # Create the outer and inner polygons
    outer_polygon = Polygon(coords)
    inner_polygon = outer_polygon.buffer(-border_width)

    # Draw the outer polygon
    draw.polygon(coords, fill=border_color)

    # Extract the coordinates of the inner polygon
    inner_coords = list(inner_polygon.exterior.coords)

    # Draw the inner polygon with the fill color
    draw.polygon(inner_coords, fill=color)

    return img


def draw_circle(img, color, border_color="black", border_width=4):
    draw = ImageDraw.Draw(img)

    # Calculate the bounding box of the circle
    min_coord = (0, 0)
    max_coord = img.size

    # Draw the circle as a black ellipse
    draw.ellipse(
        [min_coord, max_coord], outline=border_color, width=border_width, fill=color
    )

    return img


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


def draw_and_transform(
    circle,
    rectangles,
    category,
    obj_bbox=(-3, 3, -3, 3),
    instance=None,
    scene_bbox=(-3, 3, -3, 3),
    border_color="black",
    border_width=4,
    img_size=512,
    color=(255, 255, 255),
):
    xmin, xmax, ymin, ymax = scene_bbox
    scene_xy_max_size = max(xmax - xmin, ymax - ymin)

    xmin, xmax, ymin, ymax = obj_bbox
    obj_xy_max_size = max(xmax - xmin, ymax - ymin)

    world_to_img_xy_scale = img_size / scene_xy_max_size

    xy_scale = obj_xy_max_size / scene_xy_max_size

    # y is height, x is width

    if instance is not None:
        x_scale = xy_scale * instance.scale[0]
        y_scale = xy_scale * instance.scale[2]
    else:
        x_scale, y_scale = xy_scale, xy_scale

    def remap_coord(coord, scale):
        return (coord - (img_size) / 2) * scale + img_size / 2

    # vertices = mesh.vertices
    # Compute the size of the scaled rectangle
    # min_coords = np.min(vertices, axis=0)
    # max_coords = np.max(vertices, axis=0)
    if circle is not None:
        centroid, radius = circle
        # radius = radius * 0.5
        # centroid = (centroid[0] + 15, centroid[1] - 10)
        # ymin, xmin = remap_coord(centroid[1] - radius, y_scale), remap_coord(centroid[0] - radius, x_scale)
        # radius *= xy_scale
        width, height = radius * 2 * x_scale, radius * 2 * y_scale

        # if abs(centroid[0] - 255) > 2 or abs(centroid[1]-255) > 2:
        ymin, ymax = centroid[0] - radius, centroid[0] + radius
        xmin, xmax = centroid[1] - radius, centroid[1] + radius

        img = Image.new("RGB", (int(width), int(height)), (255, 255, 255))
        img = draw_circle(
            img, color=color, border_color=border_color, border_width=border_width
        )
    else:
        rects = rectangles
        if len(rects) == 1:
            ymin, xmin, height, width = rects[0]
            ymax = ymin + height
            xmax = xmin + width
            # ymin, xmin = remap_coord(ymin, y_scale), remap_coord(xmin, x_scale)

            img_scale = img_size
            width, height = width * x_scale, height * y_scale

            # Create a new image to draw the rectangle
            img = Image.new("RGB", (int(width), int(height)), (255, 255, 255))
            img = cat_config[category]["draw_function"](
                img, color=color, border_color=border_color, border_width=border_width
            )
            # draw = ImageDraw.Draw(img)
            # draw.rectangle([0, 0, width, height], outline='black', width=border_width, fill=color)
        else:

            merged_polygon = extend_and_merge_rectangles(rects)

            xmin, ymin, xmax, ymax = merged_polygon.bounds
            width, height = (xmax - xmin) * x_scale, (ymax - ymin) * y_scale

            # Shift the coordinates of the polygon
            img = Image.new("RGB", (int(width), int(height)), (255, 255, 255))
            try:
                shifted_coords = [
                    ((x - xmin) * x_scale, (y - ymin) * y_scale)
                    for x, y in merged_polygon.exterior.coords
                ]
                img = draw_polygon(
                    img,
                    shifted_coords,
                    color=color,
                    border_color=border_color,
                    border_width=border_width,
                )
            except:
                img = draw_rectangle(
                    img,
                    color=color,
                    border_color=border_color,
                    border_width=border_width,
                )

            # ymin, xmin = remap_coord(ymin, y_scale), remap_coord(xmin, x_scale)

            # img = Image.new("RGB", (int(width), int(height)), (255, 255, 255))
            # draw = ImageDraw.Draw(img)
            # for rect in rects:
            #    yymin, xxmin, height, width = rect
            #    min_coord = (xxmin-xmin, yymin-ymin)
            #    max_coord = (xxmin-xmin+width, yymin-ymin+height)
            #    draw.rectangle([min_coord, max_coord], outline=border_color, width=border_width, fill=color)

            # img = Image.new("RGB", (int(width), int(height)), (0, 0, 0))
            # for rect in rects:
            #    ymin, xmin, height, width = rect
            #    img_scale = img_size

            #    if instance is not None:
            #        raise NotImplementedError
            #        # Scale the rectangle
            #        width *= instance.scale[0] * xy_scale
            #        height *= instance.scale[2] * xy_scale

            #    # Create a new image to draw the rectangle
            #    img = draw_rectangle(img, color=color, border_color=border_color, border_width=border_width)
            #    #draw = ImageDraw.Draw(img)
            #    #draw.rectangle([0, 0, width, height], outline='black', width=border_width, fill=color)

    xcenter = (xmin + xmax) / 2
    ycenter = (ymin + ymax) / 2
    if abs(xcenter - 255.5) > 3 or abs(ycenter - 255.5) > 3:
        xmin, xmax = remap_coord(xmin, x_scale), remap_coord(xmax, x_scale)
        ymin, ymax = remap_coord(ymin, y_scale), remap_coord(ymax, y_scale)
        xcenter = (xmin + xmax) / 2
        ycenter = (ymin + ymax) / 2
        xcoffset = xcenter - 255.5
        ycoffset = ycenter - 255.5
        if xcoffset > 0:  # expand on negative x
            bbox_xmin, bbox_xmax = math.floor(xmin - xcoffset * 2), math.ceil(xmax)
        else:
            bbox_xmin, bbox_xmax = math.floor(xmin), math.ceil(xmax - xcoffset * 2)

        if ycoffset > 0:  # eypand on negative y
            bbox_ymin, bbox_ymax = math.floor(ymin - ycoffset * 2), math.ceil(ymax)
        else:
            bbox_ymin, bbox_ymax = math.floor(ymin), math.ceil(ymax - ycoffset * 2)
        # print(xmin, xmax, bbox_xmin, bbox_xmax)
        bbox_width, bbox_height = bbox_xmax - bbox_xmin, bbox_ymax - bbox_ymin
        paste_position = (int(xmin - bbox_xmin), int(ymin - bbox_ymin))
        # print(paste_position)
        # print(width, height, bbox_width, bbox_height)

        bbox_img = Image.new("RGB", (bbox_width, bbox_height), (255, 255, 255))
        bbox_img.paste(img, paste_position)
        # img.show()
        # bbox_img.show()
        img = bbox_img
        # input()

    # bbox_img = Image.new("RGB", (obj_bbox_width, obj_bbox_height), (255,255,255))

    if instance is not None:
        angle = z_angle(instance.rot) * 180 / np.pi % 360

        # Snap the angle to one of the four cardinal directions if it's within 5 degrees
        for cardinal_angle in [0, 90, 180, 270, 360]:
            if abs(angle - cardinal_angle) <= 5:
                angle = cardinal_angle
                break
        if angle == 360:
            angle == 0

        if angle != 0:
            img = img.rotate(
                angle, resample=Image.BICUBIC, expand=True, fillcolor="white"
            )

    if instance is not None:
        xshift = instance.pos[0]
        yshift = instance.pos[2]
    else:
        xshift = 0
        yshift = 0

    # xshift += center_obj_xshift
    # yshift += center_obj_yshift
    # position = np.around((xshift * xy_scale + img_size/2, yshift * xy_scale + img_size/2)).astype(int)

    img_width, img_height = img.size
    # print(img_width/2, xmin)
    # print(img_height/2, ymin)
    # input()

    # top_left_x = int(xmin)
    # top_left_y = int(ymin)
    # print(width, height, xmin, ymin)

    # Assume original model is always centered for now
    # Apparently.. we cannot
    position = np.around(
        (
            xshift * world_to_img_xy_scale + img_size / 2,
            yshift * world_to_img_xy_scale + img_size / 2,
        )
    ).astype(int)

    img_width, img_height = img.size
    top_left_x = position[0] - img_width // 2
    top_left_y = position[1] - img_height // 2
    paste_position = (top_left_x, top_left_y)

    # top_left_x = int(xmin + xshift * world_to_img_xy_scale)
    # top_left_y = int(ymin + yshift * world_to_img_xy_scale)

    # paste_position = (top_left_x, top_left_y)
    # print(img_width, img_height, paste_position, angle)

    final_img = Image.new("RGB", (img_size, img_size), (255, 255, 255))
    final_img.paste(img, paste_position)

    return final_img


def threshold_image(image, threshold=0.01):
    binary_image = np.zeros_like(image, dtype=np.uint8)
    binary_image[image > threshold] = 255
    return binary_image


# def draw_contour(contour):
#    # Create a new 512x512 image
#    img = Image.new('RGB', (512, 512), (255, 255, 255))
#
#    # Create an ImageDraw object
#    draw = ImageDraw.Draw(img)
#
#    # Convert the contour points to a list of tuples
#    contour_points = [tuple(point) for point in contour.squeeze()]
#
#    # Draw the contour on the image
#    draw.polygon(contour_points, outline=(0, 0, 0))
#
#    return img


def find_contours(binary_image):
    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return contours


def draw_contour(contours):
    # Create a new 512x512 image
    img = Image.new("RGB", (512, 512), (255, 255, 255))

    # Create an ImageDraw object
    draw = ImageDraw.Draw(img)

    for contour in contours:
        # Convert the contour points to a list of tuples
        contour_points = [tuple(point) for point in contour.squeeze()]

        # Draw the contour on the image
        draw.polygon(contour_points, outline=(0, 0, 0), fill="black")

    return img


def visualize_rectangles(rectangles, img_size=512, fill="white"):
    img = Image.new("1", (img_size, img_size), "white")
    draw = ImageDraw.Draw(img)

    for rect in rectangles:
        row, col, h, w = rect
        draw.rectangle([(col, row), (col + w, row + h)], fill=fill, outline="black")

    return img


def visualize_circle(centroid, radius, image_size=(512, 512)):
    img = Image.new(mode="L", size=image_size, color=255)  # Create a white image
    draw = ImageDraw.Draw(img)

    # Calculate the bounding box of the circle
    top_left = (centroid[1] - radius, centroid[0] - radius)
    bottom_right = (centroid[1] + radius, centroid[0] + radius)

    # Draw the circle as a black ellipse
    draw.ellipse([top_left, bottom_right], fill="white", outline="black")

    return img


def find_largest_rectangle(
    binary_map, area_threshold=0.10, original_area=None, threshold=0.95
):
    if original_area is None:
        original_area = np.sum(binary_map == 0)

    height, width = binary_map.shape
    min_area = int(original_area * area_threshold)
    min_square_area = 1
    square_size = int(np.sqrt(min_square_area))

    step = max(2, max(height, width) // 10)

    expand_step = min(5, max(1, max(height, width) // 20))

    def is_valid_expansion(row, col, h, w):
        expansion_area = h * w
        furniture_pixels = np.sum(binary_map[row : row + h, col : col + w] == 255)
        return furniture_pixels / expansion_area >= threshold

    def check_percentage(row, col, h, w):
        expansion_area = h * w
        furniture_pixels = np.sum(binary_map[row : row + h, col : col + w] == 255)
        return furniture_pixels / expansion_area

    def expand_rectangle(row, col, h, w):
        expansions = []

        if row > expand_step:
            expansions.append((row - expand_step, col, h + expand_step, w))
        if col > expand_step:
            expansions.append((row, col - expand_step, h, w + expand_step))
        if row + h + expand_step < height:
            expansions.append((row, col, h + expand_step, w))
        if col + w + expand_step < width:
            expansions.append((row, col, h, w + expand_step))

        expansion_with_percentage = [
            (r, c, nh, nw, check_percentage(r, c, nh, nw))
            for r, c, nh, nw in expansions
        ]
        valid_expansions = [v for v in expansion_with_percentage if v[4] >= threshold]
        # valid_expansions = [(r, c, nh, nw) for r, c, nh, nw in expansions if is_valid_expansion(r, c, nh, nw)]

        if not valid_expansions:
            return row, col, h, w
        best_expansion = max(
            valid_expansions, key=lambda x: x[4] * 10000 - abs(x[2] - x[3])
        )
        return expand_rectangle(*best_expansion[0:4])

    best_rect = None
    best_area = min_area
    for row in range(0, height - square_size + 1, step):
        for col in range(0, width - square_size + 1, step):
            # print(row, col)
            if check_percentage(row, col, square_size, square_size) > 0.99:
                rect = expand_rectangle(row, col, square_size, square_size)
                _, _, h, w = rect
                area = h * w
                if area >= best_area:
                    best_area = area
                    best_rect = rect
    return best_rect


def fit_rectangles(
    binary_map, area_threshold=0.1, threshold=0.95, max_rectangles=4, min_coverage=0.6
):
    original_area = np.sum(binary_map == 0)
    # if original_area > 10000:
    #    threshold = max(threshold, 0.97)
    # if original_area > 20000:
    #    threshold = max(threshold, 0.98)
    rectangles = []

    finished_area = 0

    def subtract_rectangle(rect):
        row, col, h, w = rect
        binary_map[row : row + h, col : col + w] = 255

    original_box = None

    while True:
        labeled_components = label(binary_map == 0, connectivity=2)
        if original_box is None:
            original_box = regionprops(labeled_components)[0].bbox

        large_components = [
            region
            for region in regionprops(labeled_components)
            if region.area / original_area >= area_threshold
        ]

        if not large_components:
            break

        subtracted = False

        for component in large_components:
            min_row, min_col, max_row, max_col = component.bbox
            cropped_labeled_component = labeled_components[
                min_row:max_row, min_col:max_col
            ]
            cropped_binary_map = (
                (cropped_labeled_component == component.label) * 255
            ).astype(np.uint8)
            rect = find_largest_rectangle(
                cropped_binary_map, area_threshold, original_area, threshold
            )
            if rect is not None:
                row, col, h, w = rect
                finished_area += h * w
                global_rect = (min_row + row, min_col + col, h, w)
                rectangles.append(global_rect)
                subtract_rectangle(global_rect)
                subtracted = True
                break

        if len(rectangles) >= max_rectangles:
            break

        if not subtracted:
            if finished_area / original_area < min_coverage:
                min_row, min_col, max_row, max_col = original_box
                return [(min_row, min_col, max_row - min_row, max_col - min_col)]
            else:
                return rectangles

    if finished_area / original_area < min_coverage:
        min_row, min_col, max_row, max_col = original_box
        return [(min_row, min_col, max_row - min_row, max_col - min_col)]
    else:
        return rectangles


def extend_and_merge_rectangles(rects):
    # Convert rectangles to shapely polygons
    polygons = [
        box(rect[1], rect[0], rect[1] + rect[3], rect[0] + rect[2]) for rect in rects
    ]

    # Sort the polygons by area
    polygons = sorted(polygons, key=lambda p: p.area)

    # Helper function to check the relationship between line projections
    def compare_projections(proj1, proj2):
        min1, max1 = proj1
        min2, max2 = proj2

        if max1 < min2:
            return -1  # min_side
        elif max2 < min1:
            return 1  # max_side
        else:
            return 0  # overlapping

    # Extend non-overlapping rectangles
    for i, poly in enumerate(polygons):
        if any(
            poly.touches(other_poly) for other_poly in polygons if other_poly != poly
        ):
            continue

        min_area_change = float("inf")
        best_poly = poly
        for other_poly in polygons:
            if poly != other_poly:
                x_proj1 = (poly.bounds[0], poly.bounds[2])
                x_proj2 = (other_poly.bounds[0], other_poly.bounds[2])
                y_proj1 = (poly.bounds[1], poly.bounds[3])
                y_proj2 = (other_poly.bounds[1], other_poly.bounds[3])

                x_rel = compare_projections(x_proj1, x_proj2)
                y_rel = compare_projections(y_proj1, y_proj2)

                new_bounds = list(poly.bounds)

                if x_rel == 0 and y_rel != 0:
                    if y_rel == -1:
                        new_bounds[3] = other_poly.bounds[1]
                    else:
                        new_bounds[1] = other_poly.bounds[3]
                elif y_rel == 0 and x_rel != 0:
                    if x_rel == -1:
                        new_bounds[2] = other_poly.bounds[0]
                    else:
                        new_bounds[0] = other_poly.bounds[2]
                elif x_rel != 0 and y_rel != 0:
                    if x_rel == -1:
                        new_bounds[2] = other_poly.bounds[0] + 3
                    else:
                        new_bounds[0] = other_poly.bounds[2] - 3

                    if y_rel == -1:
                        new_bounds[3] = other_poly.bounds[1] + 3
                    else:
                        new_bounds[1] = other_poly.bounds[3] - 3

                extended_poly = box(
                    new_bounds[0], new_bounds[1], new_bounds[2], new_bounds[3]
                )
                area_change = extended_poly.area - poly.area

                if area_change < min_area_change:
                    min_area_change = area_change
                    best_poly = extended_poly

        polygons[i] = best_poly

    threshold = 3

    def is_almost_colinear(coord1, coord2, threshold):
        return abs(coord1 - coord2) <= threshold and coord1 != coord2

    changes_made = True
    while changes_made:
        changes_made = False
        for i, poly in enumerate(polygons):
            for other_poly in polygons:
                if poly != other_poly and poly.touches(other_poly):
                    new_bounds = list(poly.bounds)
                    made_change = False
                    for k in range(4):
                        if is_almost_colinear(
                            poly.bounds[k], other_poly.bounds[k], threshold
                        ):
                            if k < 2:  # min value (xmin or ymin)
                                new_bounds[k] = min(
                                    poly.bounds[k], other_poly.bounds[k]
                                )
                            else:  # max value (xmax or ymax)
                                new_bounds[k] = max(
                                    poly.bounds[k], other_poly.bounds[k]
                                )
                            made_change = True
                    if made_change:
                        changes_made = True
                        polygons[i] = box(*new_bounds)

    # for i, poly in enumerate(polygons):
    #    touches = False
    #    for other_poly in polygons:
    #        if poly != other_poly and poly.touches(other_poly):
    #            touches = True
    #    print(touches)

    # Merge all polygons into a single polygon
    merged_polygon = unary_union(polygons)
    # print(merged_polygon.bounds)

    return merged_polygon


def fit_circle(binary_map, furniture_threshold=0.90, circle_threshold=0.98):
    labeled_map = label(binary_map == 0, connectivity=2)
    if np.max(labeled_map) == 0:
        return False, None, None
    region = regionprops(labeled_map)[0]
    min_row, min_col, max_row, max_col = region.bbox
    centroid = region.centroid
    area = region.area

    # Calculate min and max radii
    min_radius = 0
    max_radius = np.sqrt((max_row - min_row) ** 2 + (max_col - min_col) ** 2) / 2

    # Binary search to find the optimal radius
    while max_radius - min_radius > 1e-6:
        mid_radius = (min_radius + max_radius) / 2

        # Create a binary mask of the circle
        rows, cols = np.ogrid[: binary_map.shape[0], : binary_map.shape[1]]
        circle_mask = (rows - centroid[0]) ** 2 + (
            cols - centroid[1]
        ) ** 2 <= mid_radius**2

        # Calculate the number of furniture pixels inside the circle
        furniture_pixels_inside_circle = np.sum((binary_map == 0) & circle_mask)

        # Calculate the ratio of furniture pixels inside the circle to the circle area
        circle_area = np.pi * mid_radius**2
        ratio = furniture_pixels_inside_circle / circle_area

        if ratio >= circle_threshold:
            min_radius = mid_radius
        else:
            max_radius = mid_radius

    # print(furniture_pixels_inside_circle, area)
    # Check if the circle covers more than the furniture_threshold of the furniture
    if furniture_pixels_inside_circle / area >= furniture_threshold:
        return True, centroid, min_radius
    else:
        return False, None, None


def parse_furniture(model_info, reparse=True, dump=True):
    modelid = model_info["model_id"]
    info_path = f"{threedfuture_dir}/{modelid}/icon_info.pkl"

    if not reparse:
        if os.path.exists(info_path):
            # Load data from the file
            with open(info_path, "rb") as f:
                circle, rectangles, bbox, depth_img, contour_vis = pickle.load(f)
            return circle, rectangles, bbox, depth_img, contour_vis

    category = model_info["category"]
    modelid = model_info["model_id"]

    try:
        meshfile = f"{threedfuture_dir}/{modelid}/raw_model.obj"
        mesh = trimesh.load(meshfile)
    except Exception as e:
        print(e)
        if dump:
            with open(info_path, "wb") as f:
                pickle.dump(("Failed", None, None, None, None), f)

        return "Failed", None, None, None, None
        # raise("Failed to Load Mesh File")

    xmin, ymin, zmin = list(mesh.vertices.min(axis=0))
    xmax, ymax, zmax = list(mesh.vertices.max(axis=0))

    # print(xmin, ymin, zmin)
    # print(xmax, ymax, zmax)

    xz_min = min(xmin, zmin) * 1.2
    xz_max = max(xmax, zmax) * 1.2

    # bbox = [-3, 3, -3, 3, mesh.vertices.min(axis=0)[1], mesh.vertices.max(axis=0)[1]]
    bbox = [xz_min, xz_max, xz_min, xz_max, ymin * 1.05, ymax * 1.05]

    depth_img = render_mesh(mesh.vertices, mesh.faces, bbox, 512)
    binary_image = threshold_image(depth_img)
    contours = find_contours(binary_image)
    # largest_contour = max(contours, key=cv2.contourArea)
    large_enough_contours = [
        contour for contour in contours if cv2.contourArea(contour) > 100
    ]
    if len(large_enough_contours) == 0:
        large_enough_contours = max(contours, key=cv2.contourArea)

    contour_vis = draw_contour(large_enough_contours)
    bin_map = np.asarray(contour_vis.convert("L"))

    # parsing_strategy = parsing_strategies[category]
    # if parsing_strategy == "standard":
    if cat_config[category]["fit_circle"]:
        is_circle, centroid, radius = fit_circle(bin_map.copy())
    else:
        is_circle = False

    if is_circle:
        with open(info_path, "wb") as f:
            pickle.dump(((centroid, radius), None, bbox, depth_img, contour_vis), f)
        return (centroid, radius), None, bbox, depth_img, contour_vis
    else:
        rectangles = fit_rectangles(
            bin_map.copy(),
            threshold=cat_config[category]["threshold"],
            max_rectangles=cat_config[category]["max_rectangles"],
            min_coverage=cat_config[category]["min_coverage"],
        )
        with open(info_path, "wb") as f:
            pickle.dump((None, rectangles, bbox, depth_img, contour_vis), f)
        return None, rectangles, bbox, depth_img, contour_vis


if __name__ == "__main__":
    import shutil
    import pyrender
    import utils

    threedf_root_dir = "/home/achang/scenesynth/3DFRONT"
    threedfuture_dir = f"{threedf_root_dir}/3D-FUTURE-model"

    with open(f"{threedf_root_dir}/model_info.json", "r") as f:
        model_infos = json.load(f)

    cat_mapping = {
        "Desk": "desk",
        "Shelf": "wardrobe",
        "Lounge Chair / Book-chair / Computer Chair": "chair",
        "Dining Chair": "chair",
        "Wall Lamp": "lamp",
        "Couch Bed": "bed",
        "Lazy Sofa": "sofa",
        "Bunk Bed": "bed",
        "Drawer Chest / Corner cabinet": "wardrobe",
        "Chaise Longue Sofa": "sofa",
        "Children Cabinet": "wardrobe",
        "Footstool / Sofastool / Bed End Stool / Stool": "chair",
        "Folding chair": "chair",
        "L-shaped Sofa": "sofa",
        "Coffee Table": "table",
        "Bar": "none",
        "Loveseat Sofa": "sofa",
        "Shoe Cabinet": "wardrobe",
        "Two-seat Sofa": "sofa",
        "Pendant Lamp": "lamp",
        None: "none",
        "armchair": "chair",
        "Dressing Chair": "chair",
        "Ceiling Lamp": "lamp",
        "Nightstand": "stand",
        "Double Bed": "bed",
        "Wardrobe": "wardrobe",
        "Bed Frame": "bed",
        "Round End Table": "table",
        "Tea Table": "table",
        "Hanging Chair": "chair",
        "Single bed": "bed",
        "Corner/Side Table": "table",
        "King-size Bed": "bed",
        "Sideboard / Side Cabinet / Console": "wardrobe",
        "Floor Lamp": "lamp",
        "Wine Cabinet": "wardrobe",
        "U-shaped Sofa": "sofa",
        "Kids Bed": "bed",
        "Bookcase / jewelry Armoire": "wardrobe",
        "Sideboard / Side Cabinet / Console Table": "table",
        "Classic Chinese Chair": "chair",
        "Three-Seat / Multi-person sofa": "sofa",
        "Dressing Table": "table",
        "Wine Cooler": "wardrobe",
        "Lounge Chair / Cafe Chair / Office Chair": "chair",
        "Dining Table": "table",
        "TV Stand": "tv_stand",
        "Three-Seat / Multi-seat Sofa": "sofa",
        "Barstool": "chair",
    }

    cats = [
        "bed",
        "stand",
        "desk",
        "chair",
        "tv_stand",
        "wardrobe",
        "sofa",
        "table",
        "lamp",
        "none",
    ]

    colors = [
        (174, 199, 232),
        (255, 187, 120),
        (152, 223, 138),
        (255, 152, 150),
        (197, 176, 213),
        (196, 156, 148),
        (247, 182, 210),
        (199, 199, 199),
        (219, 219, 141),
        (158, 218, 229),
    ]
    with open("icon_config.json", "r") as f:
        cat_config = json.load(f)

    for furniture_item, item_data in cat_config.items():
        func_name = item_data["draw_function"]
        if "min_coverage" not in item_data:
            item_data["min_coverage"] = 0.75
        item_data["draw_function"] = globals()[func_name]

    def scale_and_pad(image, size):
        aspect_ratio = min(size[0] / image.width, size[1] / image.height)
        new_size = (int(image.width * aspect_ratio), int(image.height * aspect_ratio))
        scaled_image = image.resize(new_size, Image.ANTIALIAS)
        padded_image = ImageOps.expand(
            scaled_image,
            border=(
                (size[0] - new_size[0]) // 2,
                (size[1] - new_size[1]) // 2,
                (size[0] - new_size[0] + 1) // 2,
                (size[1] - new_size[1] + 1) // 2,
            ),
            fill=0,
        )

        return padded_image

    for model_idx, model_info in tqdm(enumerate(model_infos), total=len(model_infos)):
        if model_idx < 10386:
            # continue
            pass

        if model_idx != 12024:
            # continue
            pass

        category = model_info["category"]
        modelid = model_info["model_id"]

        # if modelid != "6ba4e3f7-1756-42c1-a8f9-496eb11cb522":
        #    continue
        # if modelid != "90b84978-5c55-46c6-b0f7-29ca232578d0":
        #    continue

        if category in cat_mapping:
            mapped_cat = cats.index(cat_mapping[category])
            color = colors[mapped_cat]
        else:
            color = (255, 255, 255)

        if cats[mapped_cat] in ["lamp", "none"]:
            continue

        # if category == "Desk":
        # if category == "Shelf": #Might want to handle hollow ones
        # if category == "Lounge Chair / Book-chair / Computer Chair":
        # if category == "Dining Chair":
        # if category == "Wall Lamp":
        # if category == "Couch Bed":
        # if category == "Lazy Sofa":
        # if category == "Bunk Bed":
        # if category == "Drawer Chest / Corner cabinet":
        # if category == "Chaise Longue Sofa":
        # if category == "Children Cabinet":
        if category == "Footstool / Sofastool / Bed End Stool / Stool":
            # if category == "Folding chair":
            # if category == "L-shaped Sofa":
            # if category == "Coffee Table":
            # if category == "Bar":
            # if category == "Loveseat Sofa":
            # if category == "Shoe Cabinet":
            # if category == "Two-seat Sofa":
            # if category == "Pendant Lamp":
            # if category is None:
            # if category == "armchair":
            # if category == "Dressing Chair":
            # if category == "Ceiling Lamp":
            # if category == "Nightstand":
            # if category == "Double Bed":
            # if category == "Wardrobe":
            # if category == "Bed Frame":
            # if category == "Round End Table":
            # if category == "Tea Table":
            # if category == "Hanging Chair":
            # if category == "Single bed":
            # if category == "Corner/Side Table":
            # if category == "King-size Bed":
            # if category == "Sideboard / Side Cabinet / Console":
            # if category == "Floor Lamp":
            # if category == "Wine Cabinet":
            # if category == "U-shaped Sofa":
            # if category == "Kids Bed":
            # if category == "Bookcase / jewelry Armoire":
            # if category == "Sideboard / Side Cabinet / Console Table":
            # if category == "Classic Chinese Chair":
            # if category == "Three-Seat / Multi-person sofa":
            # if category == "Dressing Table":
            # if category == "Wine Cooler":
            # if category == "Lounge Chair / Cafe Chair / Office Chair":
            # if category == "Dining Table":
            # if category == "TV Stand":
            # if category == "Three-Seat / Multi-seat Sofa":
            # if category == "Barstool":
            # if True:
            circle, rectangles, bbox, depth_img, contour_vis = parse_furniture(
                model_info, reparse=True
            )
            if circle == "Failed":
                print(f"Failed to load mesh for {modelid}")
                continue

            if circle is not None:
                centroid, radius = circle
                rectangles_vis = visualize_circle(centroid, radius)
            else:
                rectangles_vis = visualize_rectangles(rectangles)

            # rectangles_vis = visualize_rectangles(rectangles, fill='black')
            # rectangles_vis.show()

            depth_img = Image.fromarray(((1 - depth_img) * 255).astype("uint8"))
            # if not is_circle and len(rectangles) == 1:
            # icon = draw_and_transform(is_circle, (centroid, radius), category)
            # icon = draw_and_transform(is_circle, rectangles, category)
            icon = draw_and_transform(circle, rectangles, category)

            # depth_img_simplified = render_mesh(simplified_mesh.vertices, simplified_mesh.faces, bbox, 512)
            # depth_img_simplified = Image.fromarray(((1-depth_img_simplified)*255).astype('uint8'))
            # icon_simplified = draw_and_transform(simplified_mesh, category)

            # icon.show()
            # quit()
            # Merge the images into an RGB image
            blue_channel = Image.new("L", depth_img.size, 255)
            combined = Image.merge("RGB", (icon.convert("L"), depth_img, blue_channel))

            source_path = os.path.join(threedfuture_dir, modelid, "image.jpg")
            source_image = Image.open(source_path).convert("RGB")
            # Scale the source image to 512x512, maintaining aspect ratio and padding if necessary
            source_image_resized = scale_and_pad(source_image, (512, 512))

            combined2 = Image.merge(
                "RGB",
                (contour_vis.convert("L"), rectangles_vis.convert("L"), blue_channel),
            )

            # Create a new image to hold the 2x2 grid
            grid = Image.new("RGB", (1512, 1512))

            # Paste the images into the grid
            grid.paste(icon, (0, 0))
            grid.paste(depth_img.convert("RGB"), (512, 0))

            try:
                icon_scaled = draw_and_transform(
                    circle, rectangles, category, obj_bbox=bbox[:4]
                )
                grid.paste(icon_scaled, (1024, 0))
            except Exception as e:
                print(e)
                continue

            # grid.paste(icon_simplified, (0, 512))
            # grid.paste(depth_img_simplified.convert("RGB"), (512, 512))

            grid.paste(combined, (0, 512))
            grid.paste(source_image_resized, (512, 512))
            # grid.paste(contour_vis.convert("RGB"), (0, 1024))
            grid.paste(rectangles_vis.convert("RGB"), (0, 1024))
            grid.paste(combined2.convert("RGB"), (512, 1024))

            # grid.save(f"icon_debug/{modelid}_vis.png")
            grid.save(
                f"icon_debug/{model_idx}_{modelid}_vis_{category.replace('/','').replace(' ', '')}.png"
            )
            # icon.save(f"icon_debug/{modelid}_icon.png")

            # combined.save(f"icon_debug/{modelid}_overlay.png")
            ##destination_path = os.path.join("icon_debug", f"{modelid}_render.jpg")
            # shutil.copy(source_path, destination_path)
        else:
            continue
        filename = f"{threedfuture_dir}/{modelid}"
