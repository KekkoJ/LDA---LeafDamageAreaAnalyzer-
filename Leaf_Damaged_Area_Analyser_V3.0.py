import cv2
import numpy as np
import os
from tkinter import filedialog, messagebox
import tkinter as tk
import sys
import subprocess
import importlib

# Global variables
floodx = 0
floody = 0
pencilsize = 3
cord = [0, 0, 0, 0, 0, 0, 0, 0]
nummer = 0
a4area = 624  # area of A4 paper 624 cm^2
analyzed_images = []
filename = None
org = None
result = None
height = None
drawing_mode = None  # Modalità: "draw" o "erase"

# Function to resize with aspect ratio
def resizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA, disable_event=False):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(h)
        dim = (width, int(h * r))

# Disable mouse callback if specified
    if disable_event:
        cv2.setMouseCallback("Image", lambda *args: None)

    return cv2.resize(image, dim, interpolation=inter)

# Function to choose the points of a rectangle
def choose_points(event, x, y, flags, param):
    global cord, nummer
    if event == cv2.EVENT_LBUTTONDOWN:
        cord[nummer] = x
        nummer += 1
        cord[nummer] = y
        nummer += 1
        if nummer >= 8:
            if param is not None:
                cv2.setMouseCallback(param, lambda *args: None)
                cv2.destroyWindow(param)

# Function to order the points of a rectangle
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

# Function to transform perspective based on points
def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

# Function to filter green color in image
def greenfilter(img, greenlow):
    lower_green = np.array([0, 0, 0])  
    upper_green = np.array([greenlow, 255, 255])  
    return cv2.cvtColor(cv2.inRange(img, lower_green, upper_green), cv2.COLOR_GRAY2RGB)

# Function to calculate black area in image
def calcBlackpart(lastimg):
    global filename
    gray = cv2.cvtColor(lastimg, cv2.COLOR_BGR2GRAY)
    _, gray = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    output_filename = os.path.splitext(filename)[0] + "_analyzed.jpg"
    cv2.imwrite(output_filename, lastimg)
    cntnotblack = cv2.countNonZero(gray)
    height, width = gray.shape
    cntPixels = height * width
    cntBlackPart = (cntPixels - cntnotblack) / cntPixels
    Area = round(cntBlackPart * a4area * 10) / 10
    analyzed_images.append((filename, Area))
    messagebox.showinfo("Done", f"The black area is: {Area} cm^2. Picture saved as {output_filename}")
    return Area

drawing = False
pt1_x, pt1_y = None, None
left = False

# Function to handle drawing lines on image
def line_drawing(event, x, y, flags, param):
    """Handles drawing and erasing based on the selected mode."""
    global pt1_x, pt1_y, drawing_mode, drawing
    if event == cv2.EVENT_LBUTTONDOWN:  # Mouse click
        drawing = True
        pt1_x, pt1_y = x, y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        if drawing_mode == "draw":  # Draw mode
            cv2.line(result, (pt1_x, pt1_y), (x, y), color=(0, 0, 0), thickness=pencilsize + 2)
            pt1_x, pt1_y = x, y
        elif drawing_mode == "erase":  # Erase mode
            result[(y - (pencilsize + 1)):(y + pencilsize + 1), (x - (pencilsize + 1)):(x + pencilsize + 1)] = org[(y - (pencilsize + 1)):(y + pencilsize + 1), (x - (pencilsize + 1)):(x + pencilsize + 1)]
            pt1_x, pt1_y = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

# Function to handle flood fill in image
def flood(event, x, y, flags, param):
    global floodx, floody
    if event == cv2.EVENT_LBUTTONDOWN:
        floodx = x
        floody = y

# Function for OpenCV trackbars
def nothing(x):
    pass

# Function to fill and replace pixels
def floodreplace(img, org):
    h, w = img.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    flood_flags = 4 | (255 << 8) | cv2.FLOODFILL_FIXED_RANGE
    cv2.floodFill(img, mask, (0, 0), (255, 255, 255), (10,) * 3, (10,) * 3, flood_flags)

    for x in range(0, h):
        for y in range(0, w):
            if img[x, y, 0] > 250:
                img[x, y] = org[x, y]

    return img

# Function to set Draw and Erase mode
def set_mode(mode):
    """Set the draw or erase modee."""
    global drawing_mode
    drawing_mode = mode
    print(f"Mode set to: {mode}")
    
# Function to get unique report filename
def get_unique_report_filename(base_filename):
    count = 1
    filename = f"{base_filename}{count}.txt"
    while os.path.exists(filename):
        count += 1
        filename = f"{base_filename}{count}.txt"
    return filename

# Function to analyze and save image
def analyze_and_save_image(img, filename):
    global org, result, floodx, floody, pencilsize, height, nummer, cord

    nummer = 0
    cord = [0, 0, 0, 0, 0, 0, 0, 0]

    imgfill = None  # Initialize imgfill with None or an appropriate image

    img = resizeWithAspectRatio(img, height=int(height * 0.9), disable_event=True)
    create_named_window_centered("Image", img.shape[1], img.shape[0])
    cv2.setMouseCallback("Image", choose_points)
    cv2.imshow("Image", img)

    messagebox.showinfo("how to use:", "HOW TO USE IT: Click on the four corners of the A4 sheet, starting from the top-left corner and proceeding clockwise. Then, press any key to continue. NOTE: Standard A4 paper dimensions are 21x30 cm. ATTENTION: If you click more than 4 times, the program will shut down and you will need to restart it.")

    cv2.waitKey(0)

    pts = np.array([(cord[0], cord[1]), (cord[2], cord[3]), (cord[4], cord[5]), (cord[6], cord[7])], dtype="float32")
    warped = four_point_transform(img, pts)

    img = resizeWithAspectRatio(warped, height=int(height * 0.9))
    org = img.copy()  # Save a copy of the original image

    create_named_window_centered('greenfilter', img.shape[1], img.shape[0])
    cv2.createTrackbar('green_filter', 'greenfilter', 0, 255, nothing)
    cv2.imshow('greenfilter', img)
    messagebox.showinfo("how to use:", "HOW TO USE IT: Adjust the slider until the damaged areas are black. When done press ESC to proceed")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key to proceed
            break
        gl = cv2.getTrackbarPos('green_filter', 'greenfilter')
        greenlow = int(gl + 10)
        img2 = greenfilter(img, greenlow)
        imggreen = img2 & img  # Apply filter to current img
        cv2.imshow('greenfilter', imggreen)

    h, w = img.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    cv2.destroyWindow('greenfilter')  # Close the 'greenfilter' window before continuing

    create_named_window_centered('flood', imggreen.shape[1], imggreen.shape[0])
    cv2.setMouseCallback('flood', flood)
    cv2.imshow('flood', imggreen)
    messagebox.showinfo("how to use:", "HOW TO USE IT: Click on the black areas that have not been eaten by insects. The operation after clicking may take some time. When finished, press ESC")

    while True:
        if cv2.waitKey(1) & 0xFF == 27:
            break
        cv2.imshow('flood', imggreen)
        if floodx > 0:
            cv2.floodFill(imggreen, mask, (floodx, floody), (254, 254, 254))
            floodx = 0
            floody = 0
            imggreen = floodreplace(imggreen, org)  # Ensure you pass `org` to `floodreplace`

    cv2.destroyWindow('flood')  # Close the 'flood' window when finished with this section

    result = imggreen  # black+image
    img = result

    create_named_window_centered('test draw', img.shape[1], img.shape[0])
    cv2.createTrackbar('pensize', 'test draw', 0, 10, nothing)
    cv2.setMouseCallback('test draw', line_drawing)

    # Creation of the Tkinter window for the buttons
    tk_root = tk.Tk()
    tk_root.title("Draw/Erase Mode")
    tk_root.geometry("300x100")
    draw_button = tk.Button(tk_root, text="DRAW", command=lambda: set_mode("draw"))
    erase_button = tk.Button(tk_root, text="ERASE", command=lambda: set_mode("erase"))

    draw_button.pack(side=tk.LEFT, padx=10, pady=10)
    erase_button.pack(side=tk.LEFT, padx=10, pady=10)

    messagebox.showinfo("how to use:", "HOW TO USE IT:Click DRAW to paint the area black, and click ERASE to remove the black paint. Use the left mouse button to draw or erase.Press ESC to continue.")

    # Loop to update Tkinter while drawing with OpenCV
    while True:
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break
        pencilsize = int(cv2.getTrackbarPos('pensize', 'test draw') + 2)
        tk_root.update()
        cv2.imshow('test draw', result)
    tk_root.destroy()  # Closes the Tkinter window

    Area = calcBlackpart(result)
    print("Black area percentage:", Area)

    if imgfill is not None:
        imgfill = resizeWithAspectRatio(imgfill, height=int(height * 0.9))  # Resize the fill image again

    cv2.destroyAllWindows()  # Ensure all windows are closed

# Function to create named window centered on screen
def create_named_window_centered(window_name, width, height):
    root.update_idletasks()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2
    cv2.namedWindow(window_name)
    cv2.moveWindow(window_name, x, y)

# Function to save report to file
def save_report_to_file(report_filename):
    if analyzed_images:
        with open(report_filename, "w") as report_file:
            for filename, area in analyzed_images:
                report_file.write(f"{filename}: {area} cm^2\n")
        messagebox.showinfo("Report", f"Analysis report saved as {report_filename}")
    else:
        messagebox.showinfo("Report", "No images analyzed, no report generated.")

# Function to analyze and save report
def analyze_and_save_report():
    global root, height, filename
    root.filename = filedialog.askopenfilename(title="Select file", filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
    if not root.filename:
        return
    filename = root.filename
    img = cv2.imread(root.filename)
    height = root.winfo_screenheight()
    img = resizeWithAspectRatio(img, height=int(height * 0.9))
    analyze_and_save_image(img, root.filename)

# Main function to run the program
def main():
    global root
    root = tk.Tk()
    root.withdraw()
    root.update()

    while True:
        analyze_and_save_report()
        cont = messagebox.askyesno("Continue?", "Do you want to analyze another image?")
        cv2.destroyAllWindows()  # Ensure all windows are closed before continuing
        if not cont:
            break

    report_base_filename = "analysis_report"
    report_filename = get_unique_report_filename(report_base_filename)
    save_report_to_file(report_filename)

    root.mainloop()

if __name__ == "__main__":
    main()
