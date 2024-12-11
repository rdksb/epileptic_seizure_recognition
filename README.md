# Electroencephalography (EEG)-based epileptic seizure diagnosis

The project showcases the application of machine and deep learning models to classify EEG recordings as either seizure or free seizure episodes. The dataset used for this project consists of 500 single-channel EEG segments of 23.6 sec. duration. Each segment is sampled as 4097 data points and divided into 23 chunks; each containing 178 data points for 1 second. This results in a dataset with 500*23=11500 rows and 179 columns; 178 features and one label. The label y=1 means epileptic seizure, otherwise (y=2, 3, 4 or 5) no seizure. The full description of the data can be found at [kaggle](https://www.kaggle.com/datasets/harunshimanto/epileptic-seizure-recognition).
