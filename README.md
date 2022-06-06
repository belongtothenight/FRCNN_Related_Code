# FRCNN_Related_Code
The file you see here are all used for preparing frcnn training and testing files.

## Software Version
+ Windows 11
+ Blender 3.1
+ Python 3.10.4
  + pyqt5
  + lxml
  + labelimg

## List of files and their usage
### 1. Bulk rename cmd renmae.txt
This .txt file list out steps to output a txt file inside target directory.

### 2. Bulk rename jpg.xlsx
This .xlsx file contains cmd commands for bulk renaming JPEG files.

### 3. Bulk rename xml.xlsx
This .xlsx file contains cmd commands for bulk renaming XML files.

### 4. Car Batch Render Script.bat
This .bat file is used to render multiple blender project at the same time. The problem of it is unable to utilize GPU computing somehow.

### 5. Car Batch Render Script.txt
This .txt file is used to make "Car Batch Render Script.bat"

### 6. Car Batch Render Script.bat
This .bat file is used to render multiple blender project at the same time. The problem of it is unable to utilize GPU computing somehow.

### 7. Car Batch Render Script1.txt
This .txt file is used to make "Car Batch Render Script1.txt"

### 8. Format Converter xml to csv V2.py
This .py file is used to take xml files from [labelimg](https://github.com/tzutalin/labelImg) output directory and store all the coordinate data into .csv file at assigned directory.

### 9. operator_file_export_camera.py
Thie .py file is a blender script and edited with a [post](https://blender.stackexchange.com/questions/58916/script-for-save-camera-position-to-file) in Blender StackExchange. It can export camera coordinate and rotation data into .csv file if camera(empty) is selected. The coordinate is based on the world origin.

### 10. operator_file_export_camera_origincorrected.py
Thie .py file is a blender script and edited with a [post](https://blender.stackexchange.com/questions/58916/script-for-save-camera-position-to-file) in Blender StackExchange. It can export camera coordinate and rotation data into .csv file if camera(empty) is selected. The coordinate is based on user set origin. Need to make changes about export data bias before any kind of usage.

### 11. test_frcnn v7.py
This .py file is modified from [keras-frcnn](https://github.com/kbardool/Keras-frcnn). No adjustment was made about the algorithm, but the exporting files. Exported .csv file contains FRCNN labelled area data. Exported .txt file contains frcnn overall test data. (the labeleld rate calculation is not correct, but still good enough for proximate result)
#### To use:
1. Copy all the codes inside this file and replace all the original code inside "test_frcnn.py".
2. Test again in CYCU AI Console.
3. Two files ("frcnn_test_info.txt" & "target_coordinate.csv") will generate inside folder "test_result". "txt" file contains all information about the test run; "csv" file stores all the FRCNN labeled coordinate. 

If there is any question, please contact me via email "dachuan516@gmail.com".
