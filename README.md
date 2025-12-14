# AI_Drowsiness_Detection
This repository is the CSE project for Programming 1 

# REQUIREMENTS TO RUN PROJECT
First, please make sure python 3.10 enviroment is activated

Please run:
`pip install -r requirements.txt`
to download all required libraries.

The model is face_landmarker.task from mediapipe.

For update requirements.txt
# 1. Cài đặt công cụ
pip install pipreqs nbconvert

# 2. Chuyển đổi Notebook sang script
jupyter nbconvert --to script *.ipynb

# 3. Tạo requirements.txt từ các script
pipreqs . --force

# 4. Dọn dẹp các script tạm thời
rm *.py  # (Hoặc del *.py nếu dùng Windows)