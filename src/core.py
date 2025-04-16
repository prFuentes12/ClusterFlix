import kagglehub
import os
import shutil
import pandas as pd # type: ignore
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# ==========================================
# CREATE FOLDER (IF NOT EXIST)
# ==========================================

# Get the parent directory path (outside 'src')
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))

# Define the 'data' folder path in the parent directory
data_dir = os.path.join(parent_dir, 'data')

# Create the 'data' folder if it doesn't exist
if not os.path.exists(data_dir):
    os.makedirs(data_dir)  # Create folder if it doesn't exist
    print(f"Folder created: {data_dir}")
else:
    print(f"The folder already exists: {data_dir}")


# ==========================================
# DOWNLOAD DATA 
# ==========================================

# Download dataset from KaggleHub to the default directory
dataset_path = kagglehub.dataset_download("grouplens/movielens-20m-dataset")

# Move the downloaded files to the 'data' folder
downloaded_files = os.listdir(dataset_path)

# Loop through the downloaded files and move them to the 'data' folder
for file in downloaded_files:
    source_path = os.path.join(dataset_path, file)  # Source path of the file
    destination_path = os.path.join(data_dir, file)  # Destination path in the 'data' folder
    shutil.move(source_path, destination_path)  # Move file to 'data' folder

print(f"Files moved to {data_dir}")


# Load CSVs from the 'data' folder
rating_df = pd.read_csv(os.path.join(data_dir, "rating.csv")) 
movie_df = pd.read_csv(os.path.join(data_dir, "movie.csv"))    

#Combined CSV on movieId
combined_df = rating_df.merge(movie_df, on="movieId", how="left")

# Due to memory overflow issues when creating a full user-movie matrix,
# we filter out only the most rated movies to reduce the number of columns
# and avoid exceeding memory limits during the pivot operation.
popular_movies = combined_df.groupby('title').filter(lambda x: len(x) > 50)

user_movie_matrix = popular_movies.pivot_table(index='userId', columns='title', values='rating').fillna(0)

print(user_movie_matrix)


# ==========================================
# KMeans , StandardScaler
# ==========================================

scaler = StandardScaler()
user_movie_scaled = scaler.fit_transform(user_movie_matrix)


inertia = []
k_values = range(1, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(user_movie_scaled)  
    inertia.append(kmeans.inertia_)

plt.plot(k_values, inertia, marker='o')
plt.title('Elbow Method For Optimal K')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()


k = 5 
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(user_movie_scaled)

# Añadimos los clusters a la tabla
user_movie_matrix['cluster'] = clusters


# ==========================================
# Visualization of user clusters using PCA 
# ==========================================


# Reduce dimensionality to 2 components for visualization
pca = PCA(n_components=2)
components = pca.fit_transform(user_movie_scaled)

# Plot users in the 2D PCA space, colored by their cluster
plt.scatter(components[:, 0], components[:, 1], c=clusters, cmap='rainbow')
plt.title('User Clusters Based on Movie Preferences')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.show()


# ==========================================
# Testing
# ==========================================


# Select a sample user (e.g., userId 1)
cluster_id = user_movie_matrix.loc[1, 'cluster']

# Get all users who belong to the same cluster
same_cluster_users = user_movie_matrix[user_movie_matrix['cluster'] == cluster_id].drop('cluster', axis=1)

# Calculate the average rating for each movie within the cluster
mean_ratings = same_cluster_users.mean().sort_values(ascending=False)

# Identify movies the user hasn't rated yet
user_seen = user_movie_matrix.loc[1][user_movie_matrix.loc[1] > 0].index
recommended = mean_ratings[~mean_ratings.index.isin(user_seen)]

# Display top 10 recommended movies for the user
print("🎬 Recommendations for user 1:")
print(recommended.head(10))