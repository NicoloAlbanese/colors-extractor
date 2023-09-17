import cv2
from PIL import Image
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import requests
import json
from urllib.parse import urlparse
from io import BytesIO


class ColorAnalyzer:
    def __init__(self, url_or_path, num_clusters=5, scaling_factor=10):
        self.url_or_path = url_or_path
        self.num_clusters = num_clusters
        self.scaling_factor = scaling_factor
        self.image = self.load_image()
        self.pixels, self.image_rgb = self.preprocess_image()
        self.centroids, self.percentages, self.labels = self.find_clusters()
        self.sorted_colors, self.sorted_percentages = self.sort_clusters_by_size()


    def load_image(self):
        '''
        Load the image into a 2D array 
        from the local path or URL and resize it.
        '''

        # If the input image path is a URL 
        if self.is_url():
            
            # Get the response
            response = requests.get(self.url_or_path)
            
            # If there is a problem in getting the response..
            if response.status_code != 200:
                
                # ..raise an exception
                raise Exception('URL does not exist or it is broken')
            
            # Try to extract the image from the URL
            try:
                
                # Get PIL image object file from the response 
                image = Image.open(BytesIO(response.content))
                
                # Convert image from PIL to OpenCV format
                image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # If an error occurs in processing the URL..
            except:
                
                # ..raise an exception
                raise Exception('URL may not contain an image.')
        
        # If the input image path is not a URL
        else:
            
            # Try to load the image
            try:
                
                # Load the image from a local path
                image = cv2.imread(self.url_or_path)
            
            # If there is a problem reading the local path..
            except:
                
                # ..raise an exception
                raise Exception('Invalid image path')
        
        # return the loaded image
        return image


    def is_url(self):
        '''
        Check if the input path is URL.
        '''
        # Return True if the path is a URL, False otherwise
        return 'http' in urlparse(self.url_or_path).scheme

    
    def preprocess_image(self):
        '''
        Resize the image to make processing faster.
        
        Out:
            - pixels: resized image as 2D array for clustering
            - image_rgb: resized image for plotting
        '''
        # Resize the image by the scaling factor for performances
        width = int(self.image.shape[1] * self.scaling_factor / 100)
        height = int(self.image.shape[0] * self.scaling_factor / 100)
        resized_img = cv2.resize(
            self.image, (width, height), interpolation=cv2.INTER_AREA)
        
        # Convert the image back to RGB
        image_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        
        # Extract pixels as 2D array for clustering
        pixels = image_rgb.reshape(-1, 3)
        
        # Return array for clustering and image for plotting
        return pixels, image_rgb

    
    def find_clusters(self):
        '''
        Find predominant colors through clustering.
        
        Out:
            - centroids: centers of the clusters (predominant colors)
            - percentages: percentage of pixels per cluster
            - labels: labels of each point            
        '''
        # Get 2D array
        pixels, _ = self.preprocess_image()

        # Instantiate clustering model
        kmeans = KMeans(n_clusters=self.num_clusters, n_init=10)
        
        # Fit the model on the image and get labels
        labels = kmeans.fit_predict(pixels)
        
        # Get centroids (predominant colors)
        centroids = kmeans.cluster_centers_.round(0).astype(int)
        
        # Get percentage of pixels belonging to each cluster
        percentages = np.bincount(labels) / len(pixels) * 100

        # Return:
        #   - centroids 
        #   - percentage of pixels per cluster        
        #   - labels of each point
        return centroids, percentages, labels
    
    
    def sort_clusters_by_size(self):
        '''
        Sort predominant colors and percentage 
        of pixels per cluster by cluster size
        in descending order.
        
        Out:
            - sorted_colors: predominant colors sorted by cluster size
            - sorted_percentages: percentages of pixels per cluster sorted by cluster size
        '''
        sorted_indices = np.argsort(self.percentages)[::-1]
        sorted_colors = self.centroids[sorted_indices]
        sorted_percentages = self.percentages[sorted_indices]
        return sorted_colors, sorted_percentages


    def plot_image(self):
        '''
        Plot the preprocessed image (resized).
        '''
        plt.imshow(self.image_rgb)
        plt.title('Preprocessed Image')
        plt.axis('off')
        plt.show()


    def plot_3d_clusters(self, width=15, height=12):
        '''
        Plot 3D visualization of the clustering.

        Args:
            - width: width of the plot
            - height: height of the plot
        '''
        # Prepare figure
        fig = plt.figure(figsize=(width, height))
        ax = fig.add_subplot(111, projection='3d')

        # Plot point labels with their cluster's color
        for label, color in zip(np.unique(self.labels), self.centroids):
            cluster_pixels = self.pixels[self.labels == label]
            r, g, b = color
            ax.scatter(cluster_pixels[:, 0], 
                       cluster_pixels[:, 1], 
                       cluster_pixels[:, 2], 
                       c=[[r/255, g/255, b/255]],  
                       label=f'Cluster {label+1}')

        # Display title, axis labels and legend
        ax.set_title('3D Cluster Visualization')
        ax.set_xlabel('r')
        ax.set_ylabel('g')
        ax.set_zlabel('b')
        plt.legend()
        plt.show()


    def plot_predominant_colors(self, width=12, height=8):
        '''
        Plot bar chart of predominant colors 
        ordered by presence in the picture.

        Args:
            - width: width of the plot
            - height: height of the plot
        '''
        # Prepare color labels for the plot
        color_labels = [f'Color {i+1}' for i in range(self.num_clusters)]
        
        # Prepare figure
        plt.figure(figsize=(width, height))
        
        # Plot bars
        bars = plt.bar(color_labels, 
                       self.sorted_percentages, 
                       color=self.sorted_colors / 255.0, 
                       edgecolor='black')

        # Add percentage of each bar on the plot
        for bar, percentage in zip(bars, 
                                   self.sorted_percentages):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(), 
                f'{percentage:.2f}%', 
                ha='center', 
                va='bottom')

        # Display title and axis labels
        plt.title(f'Top {self.num_clusters} Predominant Colors')
        plt.xlabel('Colors')
        plt.ylabel('Percentage of Pixels')
        plt.xticks(rotation=45)
        plt.show()
        
        
    def get_predominant_colors(self):
        '''
        Return a list of predominant colors.
        Each color is a JSON object with RGB code and percentage.
        '''
        # Prepare output list
        colors_json = []
        
        # For each predominant color
        for color, percentage in zip(self.sorted_colors, 
                                     self.sorted_percentages):
            # Get the RGB code
            r, g, b = color
            
            # Prepare JSON object
            color_entry = {'color': {'R': f'{r}', 
                                     'G': f'{g}', 
                                     'B': f'{b}'}, 
                           'percentage': f'{percentage:.2f}%'}
            
            # Append JSON object to color list
            colors_json.append(color_entry)

        # Return the results
        return colors_json