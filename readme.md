# CNN Image Classifier: Cats vs Dogs

1. Make sure Python is installed on your system and optionally create a virtual environment to keep dependencies organized.  
2. Install the required Python packages by running `pip install -r requirements.txt`.  
3. Prepare your dataset by placing cat and dog images into a folder named `dataset/cats_vs_dogs` with separate subfolders for cats and dogs.  
4. Optionally, create a folder called `new_images` and place a few test images inside for prediction after training.  
5. Run the main script `cnn_image_classifier.py` by executing `python cnn_image_classifier.py` in your terminal or command prompt.  
6. The script will automatically resize images, skip corrupted or unsupported files, train the CNN model, and save the trained model as `cnn_image_classifier.h5`.  
7. After training, the model will predict the class of images in the `new_images` folder and display each image with a predicted label such as Cat or Dog.    
8. You can now use the trained model to test predictions on new images and view results easily by following these steps.
