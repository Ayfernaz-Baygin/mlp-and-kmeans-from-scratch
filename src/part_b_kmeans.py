import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Ensure visuals folder exists
if not os.path.exists("visuals"):
    os.makedirs("visuals")

# Fixed seed for reproducibility
np.random.seed(42)


def minmax_fit(data):
    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)
    return data_min, data_max


def minmax_transform(data, data_min, data_max):
    denominator = data_max - data_min
    denominator = np.where(denominator == 0, 1, denominator)
    return (data - data_min) / denominator


def load_data():
    data_path = "data/midtermProject-part2-data.xlsx"
    df = pd.read_excel(data_path)

    X = df.values.astype(float)
    feature_names = list(df.columns)

    return df, X, feature_names


def initialize_centroids(X, k):
    indices = np.random.choice(X.shape[0], k, replace=False)
    return X[indices]


def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


def assign_clusters(X, centroids):
    clusters = []

    for x in X:
        distances = []
        for centroid in centroids:
            dist = euclidean_distance(x, centroid)
            distances.append(dist)

        cluster_index = np.argmin(distances)
        clusters.append(cluster_index)

    return np.array(clusters)


def update_centroids(X, clusters, k):
    new_centroids = []

    for i in range(k):
        cluster_points = X[clusters == i]

        if len(cluster_points) == 0:
            random_index = np.random.randint(0, X.shape[0])
            new_centroids.append(X[random_index])
        else:
            new_centroid = np.mean(cluster_points, axis=0)
            new_centroids.append(new_centroid)

    return np.array(new_centroids)


def kmeans(X, k, max_iter=100, tolerance=1e-6):
    centroids = initialize_centroids(X, k)

    for iteration in range(max_iter):
        clusters = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, clusters, k)

        centroid_shift = np.sqrt(np.sum((new_centroids - centroids) ** 2))

        if centroid_shift < tolerance:
            print("K-means converged at iteration:", iteration + 1)
            centroids = new_centroids
            break

        centroids = new_centroids

    return clusters, centroids


def compute_wcss(X, clusters, centroids):
    wcss = 0.0

    for i in range(X.shape[0]):
        cluster_id = clusters[i]
        centroid = centroids[cluster_id]
        wcss += euclidean_distance(X[i], centroid) ** 2

    return wcss


def compute_bcss(X, clusters, centroids):
    overall_mean = np.mean(X, axis=0)
    bcss = 0.0

    for i in range(len(centroids)):
        cluster_points = X[clusters == i]
        cluster_size = len(cluster_points)

        if cluster_size > 0:
            bcss += cluster_size * (euclidean_distance(centroids[i], overall_mean) ** 2)

    return bcss


def compute_dunn_index(X, clusters, k):
    max_intra_cluster_distance = 0.0
    min_inter_cluster_distance = float("inf")

    for cluster_id in range(k):
        cluster_points = X[clusters == cluster_id]

        if len(cluster_points) < 2:
            continue

        for i in range(len(cluster_points)):
            for j in range(i + 1, len(cluster_points)):
                dist = euclidean_distance(cluster_points[i], cluster_points[j])
                if dist > max_intra_cluster_distance:
                    max_intra_cluster_distance = dist

    for cluster_i in range(k):
        points_i = X[clusters == cluster_i]

        if len(points_i) == 0:
            continue

        for cluster_j in range(cluster_i + 1, k):
            points_j = X[clusters == cluster_j]

            if len(points_j) == 0:
                continue

            for p1 in points_i:
                for p2 in points_j:
                    dist = euclidean_distance(p1, p2)
                    if dist < min_inter_cluster_distance:
                        min_inter_cluster_distance = dist

    if max_intra_cluster_distance == 0:
        return 0.0

    return min_inter_cluster_distance / max_intra_cluster_distance


def write_results(clusters, k, wcss, bcss, dunn_index):
    with open("result.txt", "w", encoding="utf-8") as f:
        for i, cluster_id in enumerate(clusters):
            f.write("Record {}: Cluster {}\n".format(i + 1, cluster_id + 1))

        f.write("\n")

        for cluster_id in range(k):
            count = np.sum(clusters == cluster_id)
            f.write("Cluster {}: {} records\n".format(cluster_id + 1, count))

        f.write("\n")
        f.write("WCSS: {:.6f}\n".format(wcss))
        f.write("BCSS: {:.6f}\n".format(bcss))
        f.write("Dunn Index: {:.6f}\n".format(dunn_index))


def plot_single_pair(X, clusters, feature_names, x_index, y_index, show_plot=False):
    marker_list = ["o", "s", "^", "D", "x", "*", "P", "v", "<", ">"]
    color_list = ["red", "blue", "green", "purple", "orange", "brown", "pink", "gray", "olive", "cyan"]

    unique_clusters = np.unique(clusters)

    plt.figure(figsize=(8, 6))

    for cluster_id in unique_clusters:
        cluster_points = X[clusters == cluster_id]

        plt.scatter(
            cluster_points[:, x_index],
            cluster_points[:, y_index],
            label="Cluster {}".format(cluster_id + 1),
            marker=marker_list[cluster_id % len(marker_list)],
            color=color_list[cluster_id % len(color_list)],
            alpha=0.8
        )

    plt.xlabel(feature_names[x_index])
    plt.ylabel(feature_names[y_index])
    plt.title("Data / Cluster Visualization")
    plt.legend()
    plt.grid(True)

    file_name = "visuals/cluster_{}_vs_{}.png".format(
        feature_names[x_index].replace(" ", "_"),
        feature_names[y_index].replace(" ", "_")
    )

    plt.savefig(file_name, dpi=300, bbox_inches="tight")
    print("Visualization saved to:", file_name)

    if show_plot:
        plt.show()
    else:
        plt.close()


def save_required_visualizations(X, clusters, feature_names):
    required_pairs = [
        (0, 2),  # Başçevre vs Kilo
        (0, 1),  # Başçevre vs Boy
        (1, 2)   # Boy vs Kilo
    ]

    print("\nSaving required 3 visualizations...")

    for x_index, y_index in required_pairs:
        plot_single_pair(X, clusters, feature_names, x_index, y_index, show_plot=False)

    print("All required visualizations were saved into the visuals folder.")


def visualize_clusters(X, clusters, feature_names):
    print("\nData / Cluster Visualization")
    print("----------------------------")
    print("Available variables:")

    for i, feature in enumerate(feature_names):
        print("{} - {}".format(i, feature))

    try:
        x_index = int(input("Select x-axis variable index: "))
        y_index = int(input("Select y-axis variable index: "))
    except ValueError:
        print("Invalid input. Please enter integer indices.")
        return

    if x_index < 0 or x_index >= len(feature_names):
        print("Invalid x-axis index.")
        return

    if y_index < 0 or y_index >= len(feature_names):
        print("Invalid y-axis index.")
        return

    if x_index == y_index:
        print("X and Y axis cannot be the same.")
        return

    plot_single_pair(X, clusters, feature_names, x_index, y_index, show_plot=True)


def main():
    print("Loading Part B dataset...")
    df, X, feature_names = load_data()

    print("Dataset shape:", df.shape)
    print("Feature names:", feature_names)

    data_min, data_max = minmax_fit(X)
    X_norm = minmax_transform(X, data_min, data_max)

    print("\nNormalization completed.")
    print("Normalized data shape:", X_norm.shape)

    try:
        k = int(input("Enter k value: "))
    except ValueError:
        print("Invalid k value. Please enter an integer.")
        return

    if k <= 0:
        print("k must be greater than 0.")
        return

    if k > X_norm.shape[0]:
        print("k cannot be greater than the number of records.")
        return

    clusters, centroids = kmeans(X_norm, k, max_iter=100, tolerance=1e-6)

    wcss = compute_wcss(X_norm, clusters, centroids)
    bcss = compute_bcss(X_norm, clusters, centroids)
    dunn_index = compute_dunn_index(X_norm, clusters, k)

    write_results(clusters, k, wcss, bcss, dunn_index)
    save_required_visualizations(X_norm, clusters, feature_names)

    print("\nClustering completed.")
    print("Results saved to: result.txt")
    print("WCSS:", wcss)
    print("BCSS:", bcss)
    print("Dunn Index:", dunn_index)

    while True:
        print("\nMenu")
        print("1 - Data / Cluster Visualization")
        print("2 - Exit")

        choice = input("Select an option: ")

        if choice == "1":
            visualize_clusters(X_norm, clusters, feature_names)
        elif choice == "2":
            print("Program ended.")
            break
        else:
            print("Invalid option. Please try again.")


if __name__ == "__main__":
    main()