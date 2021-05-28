HOW TO USE :

1. "src" folder has all the important files :

download the EAST detection model (cannot be uploaded here because file size > 25mb) 
it will NOT work without the EAST detection model

"test_images" has the images you want to test 
	
"main.py" is the script that has to be executed

on line 357 of the main.py file , it has the image directory , you can pass in the directory of all the images 

2. Run the script after the giving the image directory of your choice

3. Now goto "output" folder which is inside "data" folder 

4. Inside the "output" folder you will find the names of the image used

5. It has segemented text of the image used and a .txt file 

6. The recognized_text.txt has the recognised text from the image put in as input 


--------- REQUIREMENTS --------------

Some Libraries with specific versions are need to make sure this runs smoothly : 
1. Python Version <= 3.5
2. Tensorflow with version <= 1.30
3. openCV is needed with version >= 4

--------- REFERENCES ----------------

1. Marti - The IAM-database: an English sentence database for offline handwriting recognition (http://www.fki.inf.unibe.ch/databases/iam-handwriting-database)

2. EAST - A text detector using openCV -  https://arxiv.org/abs/1704.03155v2

3. Overview -https://towardsdatascience.com/build-a-handwritten-text-recognition-system-using-tensorflow-2326a3487cd5 


------------ DISCLAIMER ----------------

1. If an error does occour then simply check the image path or image spelling and things like that  
2. Dont change anything else , just change the directory on line 357 of main.py
3. For visualizing , using openCV i have put boxes and the co-ordinates of all the text detected
4. THE TEXT PREDICTION IS ONLY AS GOOD AS THE MODEL I HAVE USED WHICH IS TAKEN FROM THE 3rd LINK IN MY REFERENCES SECTION






 
