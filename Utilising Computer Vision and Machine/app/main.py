from deadlift.processVideo import *
from ohp.processVideo import *
from squat.processVideo import *
from curl.processVideo import *
import tkinter as tk
from tkinter import *
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk

lift = None
camera = None
panel = None
is_recording = False
recorded_frames = []
record_button = None
submit_button = None
prev_button = None
next_button = None
start_button = None
end_button = None
process_button = None
current_frame_index = 0
start_frame_index = 0
end_frame_index = 0
bottom_frame_record = None
bottom_frame_trim = None
advice_list = None
prev_advice_button = None
next_advice_button = None
image_label = None
text_label = None
advice_label = None
score_label = None
advice_index = 0
advice_screens = []
bad_lift = False

def resizeImage(image_path, width, height):
    original_image = Image.open(image_path)
    resized_image = original_image.resize((width, height), Image.ANTIALIAS)
    return ImageTk.PhotoImage(resized_image)


# defining frequently used button styling
def styleButtons(button):
    button.config(
        relief="raised",
        bd=3,
        bg="#5f6461",
        fg="white",
        font=("Jockey One", 12),
        padx=5,
        pady=5,
        width=10,
        height=2,
        cursor="hand2",
    )

def resizeWindow(width, height):
    app.geometry(f"{width}x{height}")


def selectUpload():
    global panel
    file_path = filedialog.askopenfilename()

    if file_path:
        # open video file
        cap = cv2.VideoCapture(file_path)
        while True:
            # read a frame from the video
            ret, frame = cap.read()
            if not ret:
                break

            # convert to rgb
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # add frame to list
            recorded_frames.append(frame)
        cap.release()
        upload.place_forget()
        record.place_forget()
        instructions.place_forget()
        instructionsDemo.place_forget()
        panel = Label(app)
        panel.place(x=0, y=0)
        displayTrimMenu()
    else:
        messagebox.showwarning("Warning","Please select a file.")


def selectLift(selected_lift):
    global lift
    lift = selected_lift
    displayFileMenu()


def displayMainMenu():
    global recorded_frames, current_frame_index, start_frame_index, end_frame_index
    recorded_frames = []
    current_frame_index = 0
    start_frame_index = 0
    end_frame_index = 0

    squat.grid(row=0, column=0)
    deadlift.grid(row=0, column=1)
    ohp.grid(row=1, column=0)
    curl.grid(row=1, column=1)

    # Bind click events to the labels
    squat.bind("<Button-1>", lambda event: selectLift('squat'))
    deadlift.bind("<Button-1>", lambda event: selectLift('deadlift'))
    ohp.bind("<Button-1>", lambda event: selectLift('ohp'))
    curl.bind("<Button-1>", lambda event: selectLift('curl'))


def displayFileMenu():
    squat.grid_remove()
    deadlift.grid_remove()
    ohp.grid_remove()
    curl.grid_remove()

    upload.place(x=0, y=0)
    record.place(x=0, y=window_height / 2)
    instructions.place(x=1050, y=10)
    instructionsDemo.place(x=1050, y=300)

    upload.bind("<Button-1>", lambda event: selectUpload())
    record.bind("<Button-1>", lambda event: displayRecordingMenu())


def displayFirstFrame():
    global current_frame_index
    current_frame_index = 0
    displayFrame(current_frame_index)


def displayFrame(frame_index):
    global panel
    frame = recorded_frames[frame_index]
    # convert frame from OpenCV to PIL
    height, width, _ = frame.shape

    if height > 900 or width > 1600:

        aspect_ratio = width / height

        # calculate height and width
        if aspect_ratio > 1600 / 900:
            new_width = 1600
            new_height = int(new_width / aspect_ratio)
            frame = cv2.resize(frame, (new_width, new_height))
            resizeWindow(new_width, new_height)
        else:
            new_height = 900
            new_width = int(new_height * aspect_ratio)
            frame = cv2.resize(frame, (new_width, new_height))
            resizeWindow(new_width, new_height)
    else:
        resizeWindow(width, height)
    frame = Image.fromarray(frame)
    frame = ImageTk.PhotoImage(frame)

    # update frame in panel
    panel.configure(image=frame)
    panel.image = frame


def nextFrame():
    global current_frame_index
    if current_frame_index < len(recorded_frames) - 2:
        current_frame_index += 2
        displayFrame(current_frame_index)


def prevFrame():
    global current_frame_index
    if current_frame_index > 1:
        current_frame_index -= 2
        displayFrame(current_frame_index)


def markStart():
    global start_frame_index, current_frame_index
    start_frame_index = current_frame_index
    end_button.config(state="active")
    styleButtons(end_button)


def markEnd():
    global end_frame_index, start_frame_index, current_frame_index
    end_frame_index = current_frame_index
    process_button.config(state="active")
    styleButtons(process_button)


def closeRecordMenu():
    camera.release()
    bottom_frame_record.destroy()
    displayTrimMenu()


def displayTrimMenu():
    global prev_button, next_button, start_button, end_button, process_button, bottom_frame_trim
    displayFirstFrame()

    # define buttons for nav
    prev_button = tk.Button(app, text="Previous", command=prevFrame)
    prev_button.pack(side=tk.LEFT)
    styleButtons(prev_button)

    next_button = tk.Button(app, text="Next", command=nextFrame)
    next_button.pack(side=tk.RIGHT)
    styleButtons(next_button)

    bottom_frame_trim = tk.Frame(app)
    bottom_frame_trim.pack(side="bottom", fill="x", padx=5, pady=5)

    start_button = tk.Button(bottom_frame_trim, text="Select start of concentric", command=markStart, wraplength=100)
    styleButtons(start_button)
    start_button.pack(side="left", padx=5, pady=5, expand=True, fill="x")

    end_button = tk.Button(bottom_frame_trim, text="Select end of concentric", command=markEnd, state="disabled", wraplength=100)
    styleButtons(end_button)
    end_button.pack(side="left", padx=5, pady=5, expand=True, fill="x")

    process_button = tk.Button(bottom_frame_trim, text="Process lift", command=processLift, state="disabled", wraplength=100)
    styleButtons(process_button)
    process_button.pack(side="left", padx=5, pady=5, expand=True, fill="x")


def processLift():
    global start_frame_index, end_frame_index, recorded_frames
    if end_frame_index > start_frame_index + 20:
        recorded_frames = recorded_frames[start_frame_index:end_frame_index + 1]
        saveVideo(recorded_frames)
        if lift == 'deadlift':
            try:
                displayHelp(*processDeadlift('output.mp4'))
            except Exception as e:
                messagebox.showerror("Error","Error processing deadlift:")
        elif lift == 'squat':
            try:
                displayHelp(*processSquat('output.mp4'))
            except Exception as e:
                messagebox.showerror("Error","Error processing squat:")
        elif lift == 'ohp':
            try:
                displayHelp(*processOHP('output.mp4'))
            except Exception as e:
                messagebox.showerror("Error","Error processing overhead press:")
        elif lift == 'curl':
            try:
                displayHelp(*processCurl('output.mp4'))
            except Exception as e:
                messagebox.showerror("Error","Error processing curl:")
    else:
        messagebox.showwarning("Warning", "Please use a longer clip of at least 20 frames.")


def displayHelp(score, advice, issue_frame):
    global advice_label, advice_list, advice_screens, advice_index, bad_lift
    panel.destroy()
    prev_button.destroy()
    next_button.destroy()
    bottom_frame_trim.destroy()

    app.geometry(f"{window_width}x{window_height}")

    score = str(round(score, 1))
    score_str = "Form Score: {} / 10".format(score)

    score_label = tk.Label(app, text=score_str, bg='#444745', fg='white', font=("Jockey One", 44))
    score_label.place(x=550, y=50)
    if len(advice[0]) == 1:
        advice_label = tk.Label(app, text=advice, font=("Jockey One", 36), width=55, bg='#444745', fg='white')
        advice_label.place(x=200, y=200)

    else:
        bad_lift = True

        # array of advice + imagery
        advice_screens = []

        # iterate through zipped arrays
        for advice, frame in zip(advice, issue_frame):

            # retrieve image data as RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_image = Image.fromarray(frame)

            # resize image to fit label whilst keeping aspect ratio
            widget_width = 700
            widget_height = 700
            ratio = min(widget_width / frame_image.width, widget_height / frame_image.height)
            new_width = int(frame_image.width * ratio)
            new_height = int(frame_image.height * ratio)
            resized_image = frame_image.resize((new_width, new_height), Image.ANTIALIAS)
            frame_image = ImageTk.PhotoImage(resized_image)

            # create image label
            image_label = tk.Label(app, image=frame_image, bg="#444745")
            image_label.image = frame_image

            # create text label
            text_label = tk.Label(app, text=advice, fg='white', bg="#444745", font=("Jockey One", 30), wraplength=500)

            # append to array
            advice_screens.append([image_label,text_label])

        # place initial advice / imagery
        advice_index = 0
        advice_screens[advice_index][0].place(x=150, y=180)
        advice_screens[advice_index][1].place(x=800, y=300)

        def nextAdvice():
            # hide currently displayed advice / imagery
            global advice_index
            advice_screens[advice_index][0].place_forget()
            advice_screens[advice_index][1].place_forget()

            # increment counter and wrap with modulo
            advice_index += 1
            advice_index = advice_index % len(advice_screens)

            # display new advice / imagery
            advice_screens[advice_index][0].place(x=150, y=180)
            advice_screens[advice_index][1].place(x=800, y=300)

        def prevAdvice():
            global advice_index
            advice_screens[advice_index][0].place_forget()
            advice_screens[advice_index][1].place_forget()

            advice_index -= 1
            advice_index = advice_index % len(advice_screens)

            advice_screens[advice_index][0].place(x=150, y=180)
            advice_screens[advice_index][1].place(x=800, y=300)

        prev_advice_button = tk.Button(app, text="Previous", command=prevAdvice)
        prev_advice_button.pack(side=tk.LEFT)
        styleButtons(prev_advice_button)

        next_advice_button = tk.Button(app, text="Next", command=nextAdvice)
        next_advice_button.pack(side=tk.RIGHT)
        styleButtons(next_advice_button)

    # create label for home button
    home = tk.Label(app, text="Return to Main Menu \U0001F3E0", font=("Jockey One", 24),
                    bg='#444745', fg='white', cursor="hand2")

    def toMainMenu(event):
        global bad_lift
        home.destroy()
        score_label.destroy()

        if advice_label:
            advice_label.destroy()

        if bad_lift:
            advice_screens[advice_index][0].destroy()
            advice_screens[advice_index][1].destroy()
            next_advice_button.destroy()
            prev_advice_button.destroy()

        bad_lift = False
        displayMainMenu()

    home.bind("<Button-1>", toMainMenu)
    home.place(x=0, y=0)


def displayRecordingMenu():
    global camera, panel, record_button, submit_button, bottom_frame_record

    # create camera instance
    camera = cv2.VideoCapture(0)

    # throw warning if no camera
    if not camera.isOpened():
        messagebox.showwarning("Warning", "No camera detected.")

    else:
        # remove current menu elements
        upload.place_forget()
        record.place_forget()
        instructions.place_forget()
        instructionsDemo.place_forget()


        def toggleRecording():
            global is_recording, recorded_frames
            if not is_recording:
                recorded_frames = []
                is_recording = True
                record_button.config(text="Stop Recording", bg='red')
            else:
                is_recording = False
                record_button.config(text="Start Recording", bg='green')
                submit_button.config(state="active")
                styleButtons(submit_button)

        # update camera feed
        def updateFrame():
            ret, frame = camera.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # resize recording window to match camera resolution
                height, width, _ = frame.shape
                resizeWindow(width, height)

                img = Image.fromarray(frame)
                img = ImageTk.PhotoImage(image=img)

                # display frame as photo image on panel
                panel.config(image=img)
                panel.image = img

                # save frames if the user is recording
                if is_recording:
                    recorded_frames.append(frame)

            # continuously update video feed
            panel.after(1, updateFrame)

        # create panel to hold feed
        panel = Label(app)
        panel.place(x=0, y=0)

        bottom_frame_record = tk.Frame(app)
        bottom_frame_record.pack(side="bottom", fill="x", padx=5, pady=5)

        # start/stop recording buttons
        record_button = Button(bottom_frame_record, text="Start Recording", font=('Jockey One', 16, "bold"), command=toggleRecording)
        record_button.pack(side="left", padx=5, pady=5, expand=True, fill="x")
        styleButtons(record_button)
        record_button.config(bg='green')

        # submit button
        submit_button = Button(bottom_frame_record, text="Submit", font=('Jockey One', 16, "bold"), command=closeRecordMenu, state="disabled")
        submit_button.pack(side="left", padx=5, pady=5, expand=True, fill="x")
        styleButtons(submit_button)

        # updating from feed
        updateFrame()


def saveVideo(frames):
    if len(frames) == 0:
        print("No frames recorded.")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frames[0].shape[1], frames[0].shape[0]))

    for frame in frames:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out.write(rgb_frame)

    out.release()
    print("Video saved as 'output.mp4'.")


# create application window
app = Tk()
app.title("Form Analysis System")
app.configure(bg='#444745')
app.option_add("*Font", ('Jockey One', 12))

# set window dimensions
window_width = 1600
window_height = 900
app.geometry(f"{window_width}x{window_height}")

# resize all images
squatImage = resizeImage("assets/squat.jpg", window_width // 2, window_height // 2)
deadliftImage = resizeImage("assets/deadlift.jpg", window_width // 2, window_height // 2)
ohpImage = resizeImage("assets/ohp.jpg", window_width // 2, window_height // 2)
curlImage = resizeImage("assets/curl.jpg", window_width // 2, window_height // 2)
uploadImage = resizeImage("assets/upload.jpg", 1000, (window_height // 2) - 1)
recordImage = resizeImage("assets/record.jpg", 1000, (window_height // 2) - 1)
instructionsImage = resizeImage("assets/instructions.png", 500, 260)
instructionsDemoImage = resizeImage("assets/instructionDemo.png", 500, 600)

# define all labels
upload = Label(app, image=uploadImage, cursor="hand2")
record = Label(app, image=recordImage, cursor="hand2")
squat = Label(app, image=squatImage, cursor="hand2")
deadlift = Label(app, image=deadliftImage, cursor="hand2")
ohp = Label(app, image=ohpImage, cursor="hand2")
curl = Label(app, image=curlImage, cursor="hand2")
instructions = Label(app, image=instructionsImage, bg='#444745')
instructionsDemo = Label(app, image=instructionsDemoImage)

displayMainMenu()

app.mainloop()
