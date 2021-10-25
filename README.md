# Indonesian-Sign-Language-Recognition-using-MediaPipe

Recognize Indonesian Sign Language with MediaPipe Framework.

## Project workflow
1. Create CSV file with the header on create_csv.py
2. Capture hand coordinates and save in CSV format to coordinates.csv using extract_data.py
3. Train the model using coordinates.csv data on model.py and save the model in .pickle format.
4. Capture hand coordinates on predict.py. 
5. Predict the class with captured coordinates using the saved model on predict.py.
