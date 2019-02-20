# CISE_Senior_Design_UF_2019
#This branch has all code necessary to retrain the Inception_v3 model and test it with a given train and test set of images

Use train.sh to more easily use retrain.py
 - First arg is name of trained graph and label file
 - Second arg is the parent directory of the directories for each class
    - Each class directory must contain the images for that class (jpg,jpeg, and png)
    - The name of the directory will be the name of the class

After the model is trained you can use label.sh to run label_image.py
 - First arg is name of the trained graph and label file
 - Second arg is the parent directory of the test set
    - Should follow same structure as before, except hard-coded for a Happy and Sad subdirectory
 - Generates hout.csv and sout.csv
    - These represent the happy images vs sad image results

Finally you can run process_results.py to give you the best threshold to maximize TPR and minimize FPR and plot a ROC curve
 - Can take an optional arg to specify directory of hout.csv and sout.csv
