# Indonesian-Sign-Language-Recognition-using-MediaPipe

Recognize Indonesian Sign Language with MediaPipe Framework.

## Project work flow
1. Create csv file with the header on create_csv.py
2. Capture hand coordinates and saves in csv format to coordinates.csv using extract_data.py
3. Train the model using coordinates.csv data on model.py and save the model in .pickle format.
4. Capture hand coordinates on predict.py. 
5. Predict the class with captured coordinates using the saved model on predict.py.
