# LeafDamageAareaAnalyzer

Leaf Damage Analyzer is an interactive Python tool for analyzing images of damaged leaves (e.g., by insects), placed on a white A4 sheet.  
The program automatically calculates the damaged (“black”) area in cm², allows manual corrections using drawing or erasing tools, and generates a final report with the results of all analyzed images.

**Main Features**

- Resize images while maintaining aspect ratio.
- Interactive selection of the four corners of the A4 sheet.
- Perspective transformation to correct distortion.
- Adjustable filter to isolate damaged areas.
- Flood fill tool to automatically fill selected areas.
- Draw/Erase mode for manual corrections of black areas.
- Automatic calculation of damaged area in cm².
- Save analyzed images.
- Generate a final text report with results.

 Example Output
- image1.jpg: 45.2 cm^2
- image2.jpg: 30.7 cm^2
- image3.jpg: 52.1 cm^2

**Requirements**

Python 3.8 or higher

Python libraries:
- opencv-python
- numpy
- tkinter

Install missing libraries using:
 pip install opencv-python numpy


# How to use
- Create a folder and place the script and all images you want to analyze in it.
- Select the image file to analyze.
- Click the four corners of the A4 sheet in clockwise order starting from the top-left corner.
- Adjust the green filter until the damaged areas are correctly isolated.
- Use flood fill and Draw/Erase mode to correct any errors.
- Press ESC to proceed to the next step.
- At the end, the damaged area is calculated and the analyzed image is saved.
- After analyzing all images, the program generates a text report (analysis_reportX.txt) containing all calculated areas.

Original image
![CTRL7_4](https://github.com/user-attachments/assets/721ac269-1bb1-46c4-91a4-0604f5b19653)
Analyzed image
![CTRL7_4_analyzed](https://github.com/user-attachments/assets/99fe0fb9-9ef3-4a41-b492-dbfc14e926cb)
