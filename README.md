# Indonesian-Sign-Language-Recognition-using-MediaPipe

Recognize Indonesian Sign Language with MediaPipe Framework.

## Project workflow
1. Create empty CSV file with the header on create_csv.py
2. Capture hand coordinates and save in CSV format to coordinates.csv using extract_data.py
3. Train the model using coordinates.csv data on model.py 
4. Save the models in .pickle format.
5. Capture hand coordinates on predict.py. 
6. Predict the class with captured coordinates using the saved model on predict.py.

### Requirement
1.     Python 3.8
2.     MediaPipe
3.     \item Bahasa Pemrograman: Python 3.8
    \item IDE: PyCharm 21.1.3
    \item Kerangka kerja: MediaPipe 0.8.8
    \item Pustaka: Pandas 1.3.4, Numpy 1.21.2, OpenCV 4.5.3.56, scikit-learn 1.0
