# FRCNN_Related_Code
## !!ATTENTION!! This repo is archived, and will no longer be updated.
- Developing start in 2022.05.27 05:08
- Code version: v1.0.1
- Author: Dachuan Chen

## Description
The file you see here are all used for preparing frcnn training and testing files.

## Developing Environment
+ Windows 11
+ Blender 3.1
+ Python 3.10.4
  + pyqt5
  + lxml
  + labelimg

## Developing Environment Setup
- Windows 11
  1. Skipped.
- Blender 3.1
  1. Use the ".msi" file downloaded from [Blender](https://www.blender.org/) to install. No special modification needed.
- Python 3.10.4
  1. Download from [Python](https://www.python.org/downloads/) and open the "exe" file.
  2. Choose "Custom installation" and click "Add Python 3.X to PATH".
  3. In "Advance Options" page, make sure top 5 checkboxes are all checked.
  4. In "Setup was successful" page, click "Disable path length limit".
- pyqt5
  1. Open cmd (WINDOWS + R, type "cmd"), type "pip install pyqt5".
- lxml
  1. Open cmd (WINDOWS + R, type "cmd"), type "pip install lxml".
- labelimg
  1. Go to [Labelimg Github](https://github.com/tzutalin/labelImg), click "Code" (green button), click "Download ZIP".
  2. Extract the downloaded file in the directory that you remembered.
  3. Follow the Windows installation guide in [Labelimg Github](https://github.com/tzutalin/labelImg).
  4. Keep in mind that if the last step of installation doesn't work, try to check your directory whether it is only consist of English, delete all Python in your computer and reinstall desired version, close cmd and retry.

## List of files and their usage
### 1. Bulk rename cmd renmae.txt
This .txt file list out steps to output a txt file inside target directory.
#### To use:
1. Follow the steps inside and execute commands inside cmd.
2. Copy all the target file name inside "filename.txt".
3. Continue with 2 or 3.

### 2. Bulk rename jpg.xlsx
This .xlsx file contains cmd commands for bulk renaming JPEG files.
#### To use:
1. Paste all the copied filename into column 1 "old file name".
2. Type the desire file extension in column 2 "ext.".
3. Give files new names in column 3 "new file name". There should be full commands appear in column 5 "full command".
4. Copy entire column 5 "full command" and paste it to cmd.
5. Detail steps in [YouTube](https://www.youtube.com/watch?v=YtcvAt9RWdI&t=1s).

### 3. Bulk rename xml.xlsx
This .xlsx file contains cmd commands for bulk renaming XML files.
#### To use:
1. Paste all the copied filename into column 1 "old file name".
2. Type the desire file extension in column 2 "ext.".
3. Give files new names in column 3 "new file name". There should be full commands appear in column 5 "full command".
4. Copy entire column 5 "full command" and paste it to cmd.
5. Detail steps in [YouTube](https://www.youtube.com/watch?v=YtcvAt9RWdI&t=1s).

### 4. Car Batch Render Script.bat
This .bat file is used to render multiple blender project at the same time. The problem of it is unable to utilize GPU computing somehow.
#### To use:
1. Double-click on "Car Batch Render Script.bat"

### 5. Car Batch Render Script.txt
This .txt file is used to make "Car Batch Render Script.bat"
#### To use:
1. Change the target project names and render requirements.
2. Save this file and save as "filename.bat".

### 6. Car Batch Render Script1.bat
This .bat file is used to render multiple blender project at the same time. The problem of it is unable to utilize GPU computing somehow.
#### To use:
1. Double-click on "Car Batch Render Script1.bat"

### 7. Car Batch Render Script1.txt
This .txt file is used to make "Car Batch Render Script1.txt"
#### To use:
1. Change the target project names and render requirements.
2. Save this file and save as "filename.bat".

### 8. Format Converter xml to csv V2.py
This .py file is used to take xml files from [labelimg](https://github.com/tzutalin/labelImg) output directory and store all the coordinate data into .csv file at assigned directory.
#### To use:
1. Change path1 as the directory stores ".xml" files.
2. Change path2 as the directory you want to store ".csv" file and its filename.

### 9. operator_file_export_camera.py
Thie .py file is a blender script and edited with a [post](https://blender.stackexchange.com/questions/58916/script-for-save-camera-position-to-file) in Blender StackExchange. It can export camera coordinate and rotation data into .csv file if camera(empty) is selected. The coordinate is based on the world origin.
#### To use:
1. Open in Blender "script" tab.
2. Execute with Blender.

### 10. operator_file_export_camera_origincorrected.py
Thie .py file is a blender script and edited with a [post](https://blender.stackexchange.com/questions/58916/script-for-save-camera-position-to-file) in Blender StackExchange. It can export camera coordinate and rotation data into .csv file if camera(empty) is selected. The coordinate is based on user set origin. Need to make changes about export data bias before any kind of usage.
#### To use:
1. Open in Blender "script" tab.
2. Execute with Blender.

### 11. test_frcnn v7.py
This .py file is modified from [keras-frcnn](https://github.com/kbardool/Keras-frcnn). No adjustment was made about the algorithm, but the exporting files. Exported .csv file contains FRCNN labelled area data. Exported .txt file contains frcnn overall test data. (the labeleld rate calculation is not correct, but still good enough for proximate result)
The distance estimation function will need to be rewrite for better accuracy.
#### To use:
1. Copy all the codes inside this file and replace all the original code inside "test_frcnn.py".
2. Test again in CYCU AI Console.
3. Two files ("frcnn_test_info.txt" & "target_coordinate.csv") will generate inside folder "test_result". "txt" file contains all information about the test run; "csv" file stores all the FRCNN labeled coordinate. 

## Context
Email: dachuan516@gmail.com
