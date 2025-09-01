# LeafDamageAareaAnalyzer


Leaf Damage Analyzer is an interactive Python tool for analyzing images of damaged leaves (e.g., by insects), placed on a white A4 sheet.  
The program automatically calculates the damaged (“black”) area in cm², allows manual corrections using drawing or erasing tools, and generates a final report with the results of all analyzed images.

**Main Features**
Resize images while maintaining aspect ratio.
Interactive selection of the four corners of the A4 sheet.
Perspective transformation to correct distortion.
Adjustable filter to isolate damaged areas.
Flood fill tool to automatically fill selected areas.
Draw/Erase mode for manual corrections of black areas.
Automatic calculation of damaged area in cm².
Save analyzed images.
Generate a final text report with results.

**Requirements**
Python 3.8 or higher
Python libraries:
opencv-python
numpy
tkinter 
Install missing libraries using:
pip install opencv-python numpy
