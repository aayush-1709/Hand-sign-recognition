## Hand-Sign-Detector
A simple project used to collect image data and using teachable machine's trained model to test or check the predicted data.


## Installation Steps
1. create a conda environment with python=3.11 (`conda create -n img-recog-env python=3.11`)
2. `git clone https://github.com/aayush-1709/Hand-sign-recognition.git`
3. `cd Hand-sign-recognition`
4. `pip install -r requirements.txt`
5. collect data using `python data_collect.py` command. After just running, enter the image category name, then press `s` to capture and press `x` to stop.
6. Goto Teachable machine website -> https://teachablemachine.withgoogle.com/train/image and Upload your category wise sample/captured data and train the model and export the model zip. 
7. copy zip file in this directory and unzip it.
8. run `python data_test.py` to check the predicted output.


### If any issues contact -> samsinha172@gmail.com
