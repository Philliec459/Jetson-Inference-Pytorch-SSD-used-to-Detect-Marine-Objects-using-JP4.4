# Pytorch-SSD for Marine Objects 4.4
The objective of this project is to identify marine objects (Boat and Buoy) using the newest pytorch-ssd from @dustu-nv that exploits the capabilities of JP4.4. This marine SSD repository was originally inspired by Dusty Franklin’s (@dusty-nv) pytorch-ssd GitHub repository found at the following link:

https://github.com/dusty-nv/pytorch-ssd

Mr. Franklin now includes this pytorch-ssd in his latest jetson-inference repository found at the following link:

https://github.com/dusty-nv/jetson-inference/blob/master/docs/pytorch-ssd.md

This is a Single Shot MultiBox Detector using MobilNet. We basically followed his newest example as was documented except that we used our marine dataset and not fruit.  

We updated our jetson-inference with Dusty's latest code and installed all of the requirements. 

# Download additional images with labels:
We took advantage of the additional data that can now be obtained using his open_image_downloader.py program that can be run as shown below:

	python3 open_images_downloader.py --class-names "Boat"

However, we chose to download only 2500 images: 

	python3 open_images_downloader.py --max-images=2500 --class-names "Boat"

We are running our ssd programs out of the /jetson-inference/python/training/detection/ssd/ directory, but with the recommended install they can be run from any directory. 


The download was very simple creating a /data/train/
					/data/test/
					/data/validation/

subdirectory with assigned images and the 

    sub-train-annotations-bbox.csv
    sub-test-annotations-bbox.csv
    sub-validation-annotations-bbox.csv
    class-description-bbox.csv

comma delimited .csv files shown above. 

There are also the following .csv files with all annotations for each image:

    train-annotations-bbox.csv
    test-annotations-bbox.csv
    validation-annotations-bbox.csv

We have not used the above full annotation .csv files at this point. 

# Supplement Downloaded data with our Existing Marine Images and Labels:
Our downloaded data only had Boat and we also needed a Buoy classification. We added our marine dataset and added these data to the new downloaded data images and .csv files.

# Training
To train our marine dataset we used the command:

	python3 train_ssd.py --model-dir=models/marine --batch-size=4 --num-epochs=60

In the training populates our /models/marine/ subdirectory with a label.txt to define our classifications and mb1-ssd-Epoch-x-Loss-xxxxxxx.pth files for each Epoch in the models/marine subdirectory.  

Once the training is complete, we then ran the following command to create the ssd-mobilnet.onnx file using the following command line:

	python3 onnx_export.py --model-dir=models/marine

where the onnx file was made from the .pth file with the lowest Epoch Loss. The onnx file can be used to make our marine object detection estimates. We created an image subdirectory and loaded a series of Boat and Buoy images to see how well the program performed. 

First, 

	mkdir test_marine

then we run detectnet on the .jpg images from the image subdirectory:

	detectnet --model=models/marine/ssd-mobilenet.onnx --labels=models/marine/labels.txt \
          --input-blob=input_0 --output-cvg=scores --output-bbox=boxes \
            "images/*.jpg" test_marine

# Review new Downloaded data to Create High Quality Marine Dataset:
Detectnet did a fair job of boxing in the Boats and Buoys, but we were not completely satisfied. We reviewed each of the data images in the train, test and validation subdirectories and removed new downloaded images that were not representative from our training perspective or object details. Obviously, an image of a fleet boats taken from a plane is not the perspective that we are looking for or even images of the boat taken from the interior of the boat or people images with a boat somewhere in the background. We culled the images to create high quality set of new training/testing images that produced a new set of data with a Loss of 1.85 using just 60 Epochs.

We removed the images from the test/train/validation subdirectories, but we did not remove those images from our .csv files. The ssd training did point out that the images were missing and the data from these images were not used in the processing. 

# Example Results:

![Marine_Image](0.jpg)

![Marine_Image](9.jpg)

![Marine_Image](11.jpg)

![Marine_Image](22.jpg)

![Marine_Image](27.jpg)


The following type of command will be used to create our video object detection while at sea. 


	detectnet --model=models/fruit/ssd-mobilenet.onnx --labels=models/fruit/labels.txt \
          	--input-blob=input_0 --output-cvg=scores --output-bbox=boxes \
            	/dev/video1


# Command lines used in this pytorch-ssd processing using JP4.4:
	cd jetson-inference/python/training/detection/ssd
	wget https://nvidia.box.com/shared/static/djf5w54rjvpqocsiztzaandq1m3avr7c.pth -O models/mobilenet-v1-ssd-mp-0_675.pth
	pip3 install -v -r requirements.txt


	#python3 open_images_downloader.py --class-names "Boat"  - too many boats with labels too, good but huge

	python3 open_images_downloader.py --max-images=2500 --class-names "Boat"

	#added all of our marine Boat and Bouy data to train, test and validation sets including .csv file additions too.


### Train:
	Defaults Of Training: Balance_Data=False, Base_Net=None, Base_Net_Lr=0.001, Batch_Size=4, Checkpoint_Folder='Models/Marine', Dataset_Type='Open_Images', 	Datasets=['Data'], Debug_Steps=10, Extra_Layers_Lr=None, Freeze_Base_Net=False, Freeze_Net=False, Gamma=0.1, Lr=0.01, Mb2_Width_Mult=1.0, 		Milestones='80,100', Momentum=0.9, Net='Mb1-Ssd', Num_Epochs=10, Num_Workers=2, Pretrained_Ssd='Models/Mobilenet-V1-Ssd-Mp-0_675.Pth', Resume=None, 	Scheduler='Cosine', T_Max=100, Use_Cuda=True, Validation_Epochs=1, Weight_Decay=0.0005

	python3 train_ssd.py --model-dir=models/marine --batch-size=4 --num-epochs=60

### Create ONNX file:
	python3 onnx_export.py --model-dir=models/marine

### Review Object Detection:
	detectnet --model=models/marine/ssd-mobilenet.onnx --labels=models/marine/labels.txt \
          	--input-blob=input_0 --output-cvg=scores --output-bbox=boxes \
		    "images/boat*.jpg" test_marine

	detectnet --model=models/marine/ssd-mobilenet.onnx --labels=models/marine/labels.txt \
          	--input-blob=input_0 --output-cvg=scores --output-bbox=boxes \
            	/dev/video1

	detectnet --model=models/marine/ssd-mobilenet.onnx --labels=models/marine/labels.txt \
          --input-blob=input_0 --output-cvg=scores --output-bbox=boxes \
            ./buoy_boats.mp4

	detectnet --model=models/marine/ssd-mobilenet.onnx --labels=models/marine/labels.txt \
          --input-blob=input_0 --output-cvg=scores --output-bbox=boxes \
            ./sail.mp4

	detectnet /dev/video1




