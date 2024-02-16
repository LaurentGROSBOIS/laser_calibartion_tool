import cv2
import numpy as np
import math
import csv
import itertools

def apply_preprocessing(image_path):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Check if image was successfully loaded
    if image is None:
        print(f"Error loading image {image_path}")
        return None

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Define kernel for the morphological opening
    kernel = np.ones((5, 5), np.uint8)  # You might adjust the size of the kernel

    # Apply morphological opening (erosion followed by dilation)
    opened = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)

    # Apply morphological dilation
    #dilated = cv2.dilate(blurred, kernel, iterations=1)

    # Apply a binary threshold to get a binary image
    _, thresh = cv2.threshold(opened, 90, 255, cv2.THRESH_BINARY)

    # convert original image back to RGB
    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    return (color_image,thresh)

def find_crosses(image, thresh):
    # Find contours in the image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # List to store centers of crosses
    centers = []

    for contour in contours:

        # Calculate the bounding box of the contour
        x, y, width, height = cv2.boundingRect(contour)

        if width <= 17 and height <= 17:
            continue

        # Calculate moments for each contour
        M = cv2.moments(contour)

        # Use moments to calculate the centroid (center) of the contour
        if M["m00"] != 0:
            centerX = int(M["m10"] / M["m00"])
            centerY = int(M["m01"] / M["m00"])
            centers.append((centerX, centerY))

    return centers

def calculate_relative_positions(sorted_groups):
    """
    Calclate points relative to the center point

    Args:
    - sorted_groups: A list of grouped point containing a list (x, y) tuples representing the points.

    Returns:
    - A list of grouped point containing a list (x, y) tuples representing the points.
    """
    #retrieve center position
    if((len(sorted_groups) % 2 )==0):
        print("error: goup count must be odd")
        return
    center_group = sorted_groups[len(sorted_groups)//2]

    if((len(center_group) % 2 )==0):
        print("error: groups cross count must be odd")
        return
    center_pos = center_group[len(center_group)//2]

    relative_positions = []
    for group in sorted_groups:
        relative_group = [(pos[0] - center_pos[0], pos[1] - center_pos[1]) for pos in group]
        relative_positions.append(relative_group)

    return relative_positions

def display_debug_img(image, centers):
    #Display the image for debug
    for(centerX,centerY) in centers:
        # Draw a red circle at the center
        cv2.circle(image, (centerX, centerY), 25, (0, 0, 255), -1)

    # Show the image with centers marked
    scaled_image = cv2.resize(image, (0, 0), fx=0.1, fy=0.1)
    cv2.imshow("Centers", scaled_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def merge_close_points(points, min_distance=50):
    """
    Merge points that are closer than min_distance apart by averaging their positions.

    Args:
    - points: A list of (x, y) tuples representing the points.
    - min_distance: The minimum distance to consider for merging points.

    Returns:
    - A list of (x, y) tuples representing the merged points.
    """
    merged = False
    while not merged:
        merged = True  # Assume no merge is needed, prove otherwise
        i = 0
        while i < len(points):
            point1 = points[i]
            to_merge = []
            for j, point2 in enumerate(points[i+1:], start=i+1):
                distance = math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
                if distance < min_distance:
                    to_merge.append(j)
            
            if to_merge:
                # Add the original point's index to the merge list
                to_merge.append(i)
                # Calculate the average position of the points to be merged
                avg_x = sum(points[j][0] for j in to_merge) / len(to_merge)
                avg_y = sum(points[j][1] for j in to_merge) / len(to_merge)
                # Replace the first point to merge with the average position
                points[i] = (int(avg_x), int(avg_y))
                # Remove the other points that were merged
                for index in sorted(to_merge[1:], reverse=True):
                    del points[index]
                # Since we've modified the list, start over
                merged = False
                break
            else:
                i += 1  # Only increment if no merge happened
    return points

def pixels_to_mm(pixels, dpi):
    """
    Convert pixels to millimeters based on the DPI.

    Args:
    - pixels: A tuple or list of (x, y) positions in pixels.
    - dpi: The resolution of the image in dots per inch.

    Returns:
    - A tuple of (x, y) positions in millimeters.
    """
    mm_per_inch = 25.4
    return [(x / dpi * mm_per_inch, y / dpi * mm_per_inch) for x, y in pixels]


def group_by_Y(positions, y_dist=5):
    """
    Group positions within a specified radius.

    Args:
    - positions: A list of positions (x_mm, y_mm).
    - radius: The radius within which to group positions (in mm).

    Returns:
    - A list of grouped positions.
    """
    groups = []
    for pos in positions:
        found_group = False
        for group in groups:
            if any(np.abs(pos[1] - other[1]) <= y_dist for other in group):
                group.append(pos)
                found_group = True
                break
        if not found_group:
            groups.append([pos])
    return groups

def sort_and_order_groups(groups):
    """
    Sort groups by their average Y coordinate and then sort positions within each group by X coordinate.

    Args:
    - groups: A list of grouped positions.

    Returns:
    - A list of sorted and ordered groups.
    """
    # Sort each group by X coordinate
    for group in groups:
        group.sort(key=lambda pos: pos[0])

    # Sort groups by their average Y coordinate
    sorted_groups = sorted(groups, key=lambda group: np.mean([pos[1] for pos in group]))

    return sorted_groups

def export_to_csv(grouped_positions, filepath):
    """
    Export the grouped and sorted positions to a CSV file with a semicolon delimiter.

    Args:
    - grouped_positions: A list of grouped and sorted positions.
    - filepath: The path to the CSV file where data will be saved.
    """
    with open(filepath, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(['Group', 'X coordinate (mm)', 'Y coordinate (mm)'])
        for group_index, group in enumerate(grouped_positions, start=1):
            for pos in group:
                # Format the position's coordinates with commas as decimal separators
                x_str = "{:.3f}".format(pos[0]).replace('.', ',')
                y_str = "{:.3f}".format(pos[1]).replace('.', ',')
                writer.writerow([group_index, x_str, y_str])

def main(image_path, debug=False, dpi=600, csv_path='positions.csv'):
    (image,thresh) = apply_preprocessing(image_path)
    
    if thresh is None:
        return

    centers = find_crosses(image, thresh)
    filtered_centers = merge_close_points(centers, 50)
    
    positions_in_mm = pixels_to_mm(filtered_centers, dpi)

    # Sort positions by Y ascending then X ascending
    groups = group_by_Y(positions_in_mm, y_dist=5)
    sorted_groups = sort_and_order_groups(groups)
    relative_positions = calculate_relative_positions(sorted_groups)

    # Assuming the central cross has indices (0, 0)
    # Calculate indices based on sorted positions if necessary
    # This part is simplified and might need adjustment based on your specific grid alignment and sorting
    # indexed_positions = [(i, j, x_mm, y_mm) for (x_mm, y_mm), (i, j) in zip(sorted_positions, itertools.product(range(-5, 6), repeat=2))]

    # Export to CSV
    export_to_csv(relative_positions, csv_path)
    print(f"Positions have been exported to {csv_path}")

    #for group in relative_positions:
    #    print('+++++ group +++++')
    #    print("group:", group)
        
    print("element count:", len(positions_in_mm))

    if debug:
        display_debug_img(image, filtered_centers)

# Replace 'path_to_your_image.jpg' with the path to your actual image file
image_path = 'clean_pattern_600dpi-no_orig.bmp'
main(image_path, debug=True)
