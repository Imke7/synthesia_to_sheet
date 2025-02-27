import os
import re
import cv2
import yt_dlp
import numpy as np

VIDEO_FOLDER = 'cache/video'
FRAMES_FOLDER = 'cache/frames'


def extract_frames(video_name : str):
    if not os.path.exists(FRAMES_FOLDER):
        os.makedirs(FRAMES_FOLDER)

    for filename in os.listdir(FRAMES_FOLDER):
        file_path = os.path.join(FRAMES_FOLDER, filename)
        os.remove(file_path)

    cap = cv2.VideoCapture(os.path.join(VIDEO_FOLDER, video_name))

    frame_count = 0
    success, frame = cap.read()
    
    while success:
        # Save every frame
        frame_filename = os.path.join(FRAMES_FOLDER, f"frame_{frame_count:06d}.jpg")  # Padded numbers for sorting
        cv2.imwrite(frame_filename, frame)

        # Read next frame
        success, frame = cap.read()
        frame_count += 1

    cap.release()

def download_video(video_url : str):  
    # Create the folder if it does not exist yet
    if not os.path.exists(VIDEO_FOLDER):
        os.makedirs(VIDEO_FOLDER)
        
    with yt_dlp.YoutubeDL({'format': 'bv[ext=mp4]', 'noplaylist': True}) as ydl:
        info = ydl.extract_info(video_url, download=False)  
        video_title = info.get('title', 'Unknown Title')  # Get title
        sanitized_title = re.sub(r'[^a-zA-Z0-9\s]', '', video_title)  # Sanitize title
        
        # Prepare the output template with sanitized title
        outtmpl = os.path.join(VIDEO_FOLDER, f'{sanitized_title}.mp4')

        # Check if the video already exists, and if not, download it
        if not os.path.exists(outtmpl):
            ydl_opts = {
                'format': 'bv[ext=mp4]',  # Get best video-only stream in mp4 format
                'outtmpl': outtmpl,  # Use the sanitized title in the output path
                'noplaylist': True,  # Ensure it's just a single video, not a playlist
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])

    return sanitized_title + '.mp4'




def detect_keys(filename):
    # Step 1: Load the image
    image = cv2.imread(filename)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and smooth the edges
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection (adjust thresholds)
    edges = cv2.Canny(blurred, threshold1=200, threshold2=255)  # Lower threshold to catch more edges

    # Find lines using Hough Line Transform (probabilistic)
    lines = cv2.HoughLinesP(edges, 3, np.pi / 180, threshold=30, minLineLength=40, maxLineGap=20)


    # Extract and filter x-coordinates of the vertical lines
    x_coords = []
    y_coords = []

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                # Check if the line is approximately vertical (x1 == x2)
                if abs(x1 - x2) < 5:  # Allow some tolerance for verticality
                    x_coords.append(x1)
                    y_coords.append((y1, y2))

    # Calculate the median of the y-coordinates for dynamic adjustment
    median_y = np.median([min(y1, y2) for y1, y2 in y_coords])

    # Define a threshold to decide whether a line is "close enough" to another
    threshold = 150  # Adjust this threshold to your preference

    # Filter lines based on their y-coordinates and proximity to other lines
    filtered_x_coords = []

    # Go through each line and filter based on the median y-coordinate
    for i, (y1, y2) in enumerate(y_coords):
        line_y = min(y1, y2)  # Consider the upper edge of the key (or use max for bottom)
        
        # Only keep lines that are within the threshold of the median or the previous line
        if abs(line_y - median_y) < threshold:
            filtered_x_coords.append(x_coords[i])

    # Remove duplicates and sort the x-coordinates
    filtered_x_coords = sorted(set(filtered_x_coords))

    # Step two in removing close lines
    new_filtered_x_coords = []
    for i in range(len(filtered_x_coords)):
        if i == 0 or filtered_x_coords[i] - filtered_x_coords[i - 1] >= 5:
            new_filtered_x_coords.append(filtered_x_coords[i])

    filtered_x_coords = new_filtered_x_coords


    print("X coordinates of vertical lines:", filtered_x_coords)


    # Optional: Visualize the detected lines
    for x in filtered_x_coords:
        cv2.line(image, (x, 0), (x, image.shape[0]), (0, 255, 0), 2)

    # Show the result
    cv2.imshow('Detected Piano Key Separators', image)

    # Wait for 10 seconds or until a key is pressed
    cv2.waitKey(10000)  # 10000 milliseconds = 10 seconds

    # Close all OpenCV windows
    cv2.destroyAllWindows()

    return filtered_x_coords


   

if __name__ == "__main__":
    '''
    video_name = download_video("https://www.youtube.com/watch?v=tSkFpBfbUV4")
    extract_frames(video_name)
    '''


    x_coords = detect_keys(os.path.join(FRAMES_FOLDER, "frame_000000.jpg"))

