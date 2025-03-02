import os
import re
import cv2
import yt_dlp
import numpy as np
import matplotlib.pyplot as plt
from midiutil import MIDIFile

VIDEO_FOLDER = 'cache/video'
FRAMES_FOLDER = 'cache/frames'
THE_MIDI = MIDIFile(2)
ALL_KEYS = []
Y_BLACK_KEYS = -1
Y_WHITE_KEYS = -1

class PianoKey():
    def __init__(self, pitch, left_border, right_border, y_coordinate):
        self.pitch = pitch # As specified by midi format
        self.pixels = None
        self.left_border = left_border
        self.right_border = right_border
        self.y = y_coordinate
        self.active_frames = []

    def calculate_pixel_group(self):
        # compute array of four pixels in the middle of the key
        x_middle = round((self.left_border + self.right_border) / 2)
        self.pixels = [(x_middle - 1, self.y - 1), (x_middle - 1, self.y + 1), (x_middle + 1, self.y - 1), (x_middle + 1, self.y + 1)]

    def is_active():
        # Determine if the key is pressed
        pass



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


def get_user_input_notenumber():
    succes = False
    notenumber = -1
    octave = -1

    user_input = input("Welke noot is dit? (c = 0, c# = 1, d = 2, d# = 3, e = 4, f = 5, f# = 6, g = 7, g# = 8, a = 9, a# = 10, b = 11) ")
    
    while not succes:
        try:
            notenumber = int(user_input)
        except ValueError:
            print("Ongeldige input.")
        else:
            if notenumber >= 0 and notenumber <= 11:
                succes = True
            else:
                print("Ongeldige input")

    succes = False
    user_input = input("En welke octaaf? (0 t/m 8) ")

    while not succes:
        try:
            octave = int(user_input)
        except ValueError:
            print("Ongeldige input.")
        else:
            if octave >= 0 and octave <= 8:
                succes = True
            else:
                print("Ongeldige input")

    # convert to midi note number
    return octave * 12 + notenumber + 12


def is_a_sharp(midi_pitch):
    note_base = midi_pitch % 12
    sharps = [1, 3, 6, 8, 10]
    if (note_base in sharps):
        return True
    return False


def detect_keys2(filename):
    image_path = os.path.join(FRAMES_FOLDER, filename)
    image = plt.imread(image_path)

    prope_y = True 
    white_keys = True
    first_key = True

    
    def onclick(event):
        # Check if Matplotlib is in zoom mode
        if plt.get_current_fig_manager().toolbar.mode != "":
            return  # Ignore click if zoom mode is on
        
        if event.xdata is not None and event.ydata is not None:
            print(f"Geselecteerd punt: ({event.xdata}, {event.ydata})")
            plt.scatter(event.xdata, event.ydata, c='red', s=20)
            plt.draw()

            confirm = input("Is dit punt correct? (j/n): ")
            if confirm == "j" or confirm == "J":
                if prope_y and white_keys:
                    global Y_WHITE_KEYS
                    Y_WHITE_KEYS = event.ydata
                elif prope_y and not white_keys:
                    global Y_BLACK_KEYS
                    Y_BLACK_KEYS = event.ydata
                else:
                    midi_notenumber = get_user_input_notenumber()
                    if is_a_sharp(midi_notenumber):
                        y = Y_BLACK_KEYS
                    else:
                        y = Y_WHITE_KEYS
                    if first_key:
                        key = PianoKey(midi_notenumber, event.xdata, None, y)
                    else:
                        key = PianoKey(midi_notenumber, None, event.xdata, y)
                    ALL_KEYS.append(key)
                plt.close('all')
            else:
                ax.clear()
                ax.imshow(image)
                plt.draw()

    
    print("Selecteer een goede hoogte voor de witte toetsen.")
    fig, ax = plt.subplots()
    ax.imshow(image)
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    white_keys = False

    print("Selecteer een goede hoogte voor de zwarte toetsen.")
    fig, ax = plt.subplots()
    ax.imshow(image)
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    prope_y = False
    
    print("Selecteer de linker rand van de eerste volledige witte toets.")
    fig, ax = plt.subplots()
    ax.imshow(image)
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    first_key = False

    print("Selecteer de rechter rand van de laatste volledige witte toets.")
    fig, ax = plt.subplots()
    ax.imshow(image)
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

def detect_keys_debug():
    global Y_BLACK_KEYS
    global Y_WHITE_KEYS
    Y_WHITE_KEYS = 1003
    Y_BLACK_KEYS = 844
    first_key = PianoKey(43, 6.001143912643694, None, Y_WHITE_KEYS)
    last_key = PianoKey(112, None, 1889.0068388829034, Y_WHITE_KEYS)
    ALL_KEYS.append(first_key)
    ALL_KEYS.append(last_key)


def space_to_next_key(midi_pitch, size_white_key, size_black_key):
    current_key = midi_pitch % 12
    if is_a_sharp(current_key):
        return size_black_key
    
    current_key = midi_pitch_to_notename(midi_pitch)
    
    one_third = size_black_key / 3
    two_thirds = one_third * 2
    one_fourth = size_black_key / 4
    three_fourth = one_fourth * 3
    half = size_black_key / 2

    if current_key == 'c' or current_key == 'd' or current_key == 'e':
        return size_white_key - two_thirds
    if current_key == 'f' or current_key == 'b':
        return size_white_key - three_fourth
    if current_key == 'g' or current_key == 'a':
        return size_white_key - half - one_fourth
        

        
def midi_pitch_to_notename(midi_pitch):
    match midi_pitch % 12:
        case 0:
            return 'c'
        case 1:
            return 'c#'
        case 2:
            return 'd'
        case 3:
            return 'd#'
        case 4:
            return 'e'
        case 5:
            return 'f'
        case 6:
            return 'f#'
        case 7:
            return 'g'
        case 8:
            return 'g#'
        case 9:
            return 'a'
        case 10:
            return 'a#'
        case 11:
            return 'b'

def compute_key_info():
    offset = ALL_KEYS[0].left_border
    n_keys = ALL_KEYS[1].pitch - ALL_KEYS[0].pitch + 1
    last_key = ALL_KEYS.pop() # Save so the list keeps its order
    n_white_keys = 0
    for p in range(ALL_KEYS[0].pitch, last_key.pitch + 1):
        if not is_a_sharp(p):
            n_white_keys += 1
    size_white_key = (last_key.right_border - offset) / n_white_keys 
    size_black_key = 13 / 23.5 * size_white_key # 13 and 23.5 mm is standard for black and white keys respectively

    current_key = midi_pitch_to_notename(ALL_KEYS[0].pitch)
    if current_key == 'd':
        ALL_KEYS[0].left_border += size_black_key / 3
    elif current_key == 'e':
        ALL_KEYS[0].left_border += size_black_key / 3 * 2
    elif current_key == 'g':
        ALL_KEYS[0].left_border += size_black_key / 4
    elif current_key == 'a':
        ALL_KEYS[0].left_border += size_black_key / 2
    elif current_key == 'b':
        ALL_KEYS[0].left_border += size_black_key / 4 * 3

    x = ALL_KEYS[0].left_border
    x_coords = [x]
    
    x += space_to_next_key(ALL_KEYS[0].pitch, size_white_key, size_black_key) 
   
    ALL_KEYS[0].right_border = x
    x_coords.append(x)

    left_x_of_previous_C = None
    for pitch in range(ALL_KEYS[0].pitch + 1, ALL_KEYS[0].pitch + n_keys - 1):
        if pitch % 12 == 0: # correction point at c
            if left_x_of_previous_C == None:
                left_x_of_previous_C = x
            else:
                x = left_x_of_previous_C + size_white_key * 7
                left_x_of_previous_C = x
        if is_a_sharp(pitch):
            y = Y_BLACK_KEYS
        else:
            y = Y_WHITE_KEYS
        key = PianoKey(pitch, x, x + space_to_next_key(pitch, size_white_key, size_black_key), y)
        ALL_KEYS.append(key)
        # Update variables
        x += space_to_next_key(pitch, size_white_key, size_black_key)
        x_coords.append(x)
        pitch += 1

    x_coords.append(x + space_to_next_key(pitch, size_white_key, size_black_key))

    # Put the last key back
    last_key.left_border = x
    last_key.right_border = x + space_to_next_key(pitch, size_white_key, size_black_key)
    ALL_KEYS.append(last_key)

    '''
    # Visualize
    image = plt.imread(os.path.join(FRAMES_FOLDER, "frame_000000.jpg"))
    fig, ax = plt.subplots()
    ax.imshow(image)
    for x in x_coords:
        ax.axvline(x=x, color = 'red', linewidth = 1)
    plt.show()
    '''

    image = plt.imread(os.path.join(FRAMES_FOLDER, "frame_000000.jpg"))
    fig, ax = plt.subplots()
    ax.imshow(image)

    # Compute set of pixels for each key
    for key in ALL_KEYS:
        key.calculate_pixel_group()
        for coord in key.pixels:
            ax.plot(coord[0], coord[1], 'ro', markersize=1)

    #plt.show()




def safe_midi():
    # Save to file
    with open("output.mid", "wb") as f:
        THE_MIDI.writeFile(f)

def create_midi(tempo):
    THE_MIDI.addTrackName(0, 0, "Left Hand Piano")
    THE_MIDI.addTrackName(1, 0, "Right Hand Piano")
    THE_MIDI.addTempo(track = 0, time = 0, tempo = tempo)
    THE_MIDI.addTempo(track = 1, time = 0, tempo = tempo)

    # test it out
    THE_MIDI.addNote(track=0, channel=0, pitch=36, time=0, duration=1, volume=100)
    THE_MIDI.addNote(track=1, channel=0, pitch=84, time=0, duration=1, volume=100)

    safe_midi()

def get_user_metadata():
    tempo = input("Wat is het tempo? ")
    
    print("We moeten een scheiding maken van de laagste noot die bij de rechter hand hoort.")
    division_note = get_user_input_notenumber()

    

if __name__ == "__main__":
    '''
    video_name = download_video("https://www.youtube.com/watch?v=tSkFpBfbUV4")
    extract_frames(video_name)
    '''

    create_midi(120)
    detect_keys_debug()
    #detect_keys2("frame_000000.jpg")

    compute_key_info()

