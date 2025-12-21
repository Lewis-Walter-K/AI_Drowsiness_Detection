
# Real-time Drowsiness Detection System 

This project implements a low-resource real-time drowsiness detection system, unlike CNN, which extracts raw pixels from images. This system utilizes Geometric features extraction from a person’s face and posture via Mediapipe Face Mesh and Pose estimation. These extracted features are then fed into a custom Deep Neural Network (DNN) to classify the driver's current state in 3 categories:	Awake, Sleep and Yawning.

To mitigate noises caused by blinking, talking and micro-movement. A Temporial Smoothing (Rolling Average) algorithm is implemented. Making sure the predictions are reliable and precise.
## Code requirements
To deploy this project you must first git clone this project:
#### Clone project
```bash
git clone https://github.com/Lewis-Walter-K/AI_Drowsiness_Detection.git
cd AI_Drowsiness_Detection
```
#### Activate conda
You must have conda installed and activate a conda python version 3.10 enviroment
```bash
conda create -n AI python=3.10
```
Then run this to activate conda enviroment in your IDE:
```bash
conda activate AI
code
```

### LIBRARY
Install library listed in requirements.txt
```bash
pip install -r requirements.txt
```

### MODEL
Install the model "final.keras" in this drive: https://drive.google.com/drive/folders/1STUndLaRnW6XeYyLTRPyuKVHd7rDajq0?usp=drive_link


### RUN PROJECT 
```bash
python real_time_demo.py
```

## RETRAIN THE MODEL (OPTIONAL)
#### Dataset 
Install avaliable dataset from this drive:
https://drive.google.com/drive/folders/1lwmFuWgpB5Sabd7uk9gv7wSP8wOyjUck?usp=drive_link

#### COLLECT YOUR OWN DATASET (OPTIONAL)
run this python file: 
```bash
python rt_collectiondata.py
```
Choose mode: 
 * "0" For "Awake"
 * "1" For "Sleep"
 * "2" For "Yawning"
 By clicking "SPACEBAR" it will record 20 frame per second. Make the face of a person status based on the mode selected.

 ### Train your model
 After collecting your dataset -> Process your dataset
 ```bash
jupyter notebook process_data.ipynb
```
Then train your model
```bash
jupyter notebook model_Khang.ipynb
```


## Limitations
* __Lighting condition__: The system heavily relies on Mediapipe model visibility to facial and posture landmarks. Performance may degrade in low-light or infrared enviroments.
* __Eyes Blockage__: Thick eyewear or face masks may interfere with MAR and EAR calculation
* __Posture Blockage__: Scarf or too much layer surrounding the neck and upper-chest area may interfere with MAR and EAR calculation
## Demo

Insert gif or link to demo


## Authors
* __Author__: Nguyễn Thịnh Khang & Nguyễn Đình Trí 

## Reference
* https://doi.org/10.1007/978-981-16-5987-4_63
* https://doi.org/10.3390/jimaging9050091
