# AI-in-healthcare-HackUSF

AI-Driven Classification for Skin Cancer

Skin cancer is one of the most common and visually identifiable cancers. Early detection and proper classification are critical for effective treatment.
This project aims to develop an AI model that classifies skin cancer lesions using publicly available NIH image datasets and basic clinical information. The model will assist in distinguishing between benign and malignant lesions. The classification results will serve as a valuable tool for healthcare professionals in early skin cancer assessment. As a secondary aim, the model could also explore the possibility of further subcategorizing different types of skin cancer. By leveraging computer vision and machine learning, this solution provides a fast, accessible, and non-invasive tool for preliminary skin cancer assessment, making it useful for both healthcare professionals and the public.

To download the data, follow the following instructions: https://github.com/lab-rasool/MINDS

Run the following on the PDGX0002 machine. Use the docs for the MINDS project using the link above. 

1. pip install med-minds pandas
2. Copy and paste the code in the main.txt file into main.py file
3. Rename the PATH_TO_SAVE_DATA  to the path where you want to save the data, and 
4. Wherever you are running the main.py file, make a .env file and put the following on it:
HOST=127.0.0.1
PORT=5433
DB_USER=postgres
PASSWORD=minds-secure-password
DATABASE=minds
5. Run the main.py file

To download the benign and malign lesions data, please use this link: https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign/data
