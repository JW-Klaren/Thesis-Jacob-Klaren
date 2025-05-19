import pandas as pd
import numpy as np
from pyts.image import GramianAngularField
from PIL import Image
import os
import matplotlib.pyplot as plt

# Function to generate GAF images with color and larger size
def generate_gaf_images(data, num_days, output_folder, image_size=100, colormap='coolwarm'):
    gaf = GramianAngularField(image_size=num_days, method='summation')
    gaf_images = []
    labels = []

    for i in range(len(data) - num_days):
        # Extract the stock prices for the current window
        window = data[i:i+num_days]
        
        # Generate GAF image
        gaf_image = gaf.fit_transform(window.reshape(1, -1))
        gaf_image = gaf_image[0]  # Remove the batch dimension

        # Apply a colormap to the GAF image
        plt.imshow(gaf_image, cmap=colormap, interpolation='nearest')
        plt.axis('off')  # Hide axes

        # Save the GAF image with a larger size
        image_name = f"GAF{i+1}_{i+num_days}.png"
        image_path = os.path.join(output_folder, image_name)
        plt.savefig(image_path, bbox_inches='tight', pad_inches=0, dpi=image_size)
        plt.close()  # Close the plot to free memory

        # Store only the image name (without the full file path)
        gaf_images.append(image_name)
        
        # Determine the label (1 if price went up, 0 if down)
        price_change = data[i+num_days] - data[i+num_days-1]
        label = 1 if price_change > 0 else 0
        labels.append(label)
    
    return gaf_images, labels

# Load the CSV file
file_path = '/home/u247708/thesis_jwk/data/testset.csv'  # Replace with your file path
df = pd.read_csv(file_path, header=None)

# Extract the stock prices (assuming the prices are in the second column)
stock_prices = df.iloc[:, 1].values

# Number of days to include in each GAF image
num_days = 10

# Create a folder to save the images
output_folder = '/home/u247708/thesis_jwk/data/GAF_images_testset'
os.makedirs(output_folder, exist_ok=True)

# Generate GAF images and labels
gaf_images, labels = generate_gaf_images(stock_prices, num_days, output_folder, image_size=100, colormap='coolwarm')

# Create a DataFrame to store the results
results_df = pd.DataFrame({
    'GAF Image': gaf_images,  # Only the image name (e.g., "GAF1_10.png")
    'Label': labels
})

# Save the results to a new CSV file
results_df.to_csv('/home/u247708/thesis_jwk/data/gaf_labels_testset.csv', index=False)

print(f"GAF images have been saved in the folder: {output_folder}")
print("GAF labels have been saved in 'gaf_labels.csv'.")
