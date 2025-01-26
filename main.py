import cv2
import numpy as np
from sklearn.cluster import KMeans

def detect_color_grid(image_path, grid_size):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape

    cell_height = height // grid_size
    cell_width = width // grid_size
    reshaped_image = image.reshape(-1, 3)

    n_clusters = grid_size + 2  # Add extra clusters to distinguish similar colors
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(reshaped_image)
    clustered_colors = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_.reshape(height, width)

    # Map clustered colors to unique codes (C1, C2, ..., CN)
    color_map = {}
    for i, color in enumerate(clustered_colors):
        color_map[i] = f"C{i+1}"

    used_colors = set(labels.flatten())
    color_map = {k: color_map[k] for k in used_colors}
    color_grid = []

    for row in range(grid_size):
        color_row = []
        for col in range(grid_size):
            y_start, y_end = row * cell_height, (row + 1) * cell_height
            x_start, x_end = col * cell_width, (col + 1) * cell_width
            cell_labels = labels[y_start:y_end, x_start:x_end]

            unique_labels, counts = np.unique(cell_labels, return_counts=True)
            dominant_label = unique_labels[np.argmax(counts)]

            color_code = color_map[dominant_label]
            color_row.append(color_code)

        color_grid.append(color_row)

    return color_grid

image_path = 'queen3.png'
grid_size = 10
result = detect_color_grid(image_path, grid_size)
print(result)
