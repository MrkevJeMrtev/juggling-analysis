import cv2
import numpy as np
from enum import Enum
from matplotlib import pyplot as plt
import scipy.optimize
from tkinter.filedialog import askopenfilename

"""
Definition of constants
"""
class States(Enum):
    INIT = 0
    TRACKING = 1
    COLOR_PICKER = 2
    INFO = 3
    ANALYSIS = 4
    BALL_REFINEMENT = 5
    SET_HEIGHT = 6
    END = 7

def solve_for_x(y, a, b, c):
    if b**2-4*a*(c-y) >= 0:
        return [(-b+np.sqrt(b**2-4*a*(c-y)))/(2*a), (-b-np.sqrt(b**2-4*a*(c-y)))/(2*a)]

def parabola(x, a, b, c):
    return a*x**2+b*x+c

# if the window doesn't fit, change this constant
RESIZE = 0.4

# panel parameters
P_HEIGHT = 1/12
P_LT = (0, 0)
P_COLOR = (0, 0, 0)

PT_BL = 1/108
T_FONT = cv2.FONT_HERSHEY_SIMPLEX
T_SCALE = 1/23
T_COLOR = (255, 255, 255)
T_WIDTH = 3

# object panel
PO_RB = 1/12

# parameters of color reading circle
C_SIZE = 1/30
C_CIRCLE = 1/15
C_COLOR = (25, 25, 255)
C_WIDTH = 2
C_SIZE_DEV = 1/50

BLACK = np.array([[0],[0],[0]])

# Structuring element kernel size
ERO_SZ = 15

# window names
WNAME_APP = "Tracking App"
WNAME_GRAPH = "Analysis graph"
WNAME_MASK = "Detection mask"
WNAME_OBJECTS = "Detected objects"


"""
Model object class
"""
class ModelBallObject:
    def __init__(self, size, color):
        self.path = []
        self.size = size
        self.color = color
        self.change_times = []

    def generate_throw(self, start_position, end_position, flight_time, gravity, shift):
        y_factor = (0.5*gravity*(flight_time**2))/flight_time
        x_factor = (end_position[0] - start_position[0])/flight_time
        for t in range(int(flight_time-shift)+1):
            self.path.append((int(start_position[0]+x_factor*(t+shift)), int(start_position[1]-y_factor*(t+shift)+(gravity*(t+shift)**2)/2)))
        return self.path

    def generate_hold(self, start_position, end_position, dwell_time, lowest, shift):
        radius = np.abs(start_position[0] - end_position[0])/2
        center = np.average([start_position[0], end_position[0]])
        if start_position[0] < end_position[0]:
            for i in range(int(dwell_time+shift)):
                self.path.append((int(center+radius*np.cos(np.pi+(i+1-shift)*np.pi/(dwell_time+1))), int(start_position[1]+(lowest-start_position[1])*np.sin((i+1-shift)*np.pi/(dwell_time+1)))))
        else:
            for i in range(int(dwell_time+shift)):
                self.path.append((int(center-radius*np.cos(np.pi+(i+1)*np.pi/(dwell_time+1))), int(start_position[1]+(lowest-start_position[1])*np.sin((i+1)*np.pi/(dwell_time+1)))))                

"""
Object class
"""
class BallObject:
    def __init__(self, size, bgr_color = BLACK, hsv_m = BLACK, hsv_stdev = BLACK):
        self.history = []
        self.k_stdev = np.array([[1],[2],[2]])
        self.set_params(bgr_color, hsv_m, hsv_stdev, size)
        self.in_air = False
        self.parab = []
        self.last_catch = 0
        self.last_throw = 0
        self.catches = []
        self.throws = []
        self.change_times = []
        self.throwing_times = []

    def set_params(self, bgr_color, hsv_m, hsv_stdev, size):
        self.hsv_m = hsv_m
        self.hsv_stdev = hsv_stdev
        self.bgr_color = bgr_color
        self.size = size

    def __str__(self):
        bgr_str = " ".join(map(str, self.bgr_color))
        hsv_str = " ".join(map(str, self.hsv_m)) + " " + " ".join(map(str, self.hsv_stdev))
        return  f"{bgr_str} {hsv_str} {self.size}"

"""
Application class
"""
class AppState:
    def __init__(self, frame, state = States.INIT):
        self.state = state
        self.rec_paused = True
        self.object_id = 0
        self.objects = []
        self.get_frame_size(frame)
        self.circle_size = int(C_SIZE*self.width)
        self.dwell_time = []
        self.flight_time = []
        self.model_objects = []
        self.gravity = []

    def get_frame_size(self, frame):
        self.height, self.width, _ = frame.shape
        self.catch_height = self.height//2
        self.x = self.width // 2
        self.y = self.height // 2
        self.center = (self.x, self.y)

    def get_ballmask(self):
        circle_img = np.zeros((self.height, self.width), np.uint8)
        cv2.circle(circle_img, (self.x, self.y), self.circle_size, 1, -1)
        
        return circle_img

#############################################################################
# Mouse callback functions
#############################################################################
    def get_ball_center_on_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.x = int(x/RESIZE)
            self.y = int(y/RESIZE)

    def get_catch_height_on_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.catch_height = int(y/RESIZE)

#############################################################################
# functions plotting panels
#############################################################################
    def show_status(self, frame, title, desc):
        cv2.rectangle(frame, P_LT, (self.width, int(P_HEIGHT*self.height)), P_COLOR, -1)
        cv2.putText(frame, title, (int(self.height*PT_BL), int(2*self.height*P_HEIGHT/3-self.height*PT_BL)), T_FONT, int(T_SCALE*2*(self.height*P_HEIGHT/2-2*self.height*PT_BL)), T_COLOR, T_WIDTH)
        cv2.putText(frame, desc, (int(self.height*PT_BL), int(self.height*P_HEIGHT-self.height*PT_BL)), T_FONT, int(T_SCALE*(self.height*P_HEIGHT/2-2*self.height*PT_BL)), T_COLOR, T_WIDTH)

    def show_object_panel(self, frame):
        cx = int((PO_RB*self.width) // 2)

        cv2.rectangle(frame, (0, int(P_HEIGHT*self.height)), (int(PO_RB*self.width), int(P_HEIGHT*self.height+len(self.objects)*PO_RB*self.width)), P_COLOR, -1)
        for i in range(len(self.objects)):
            cv2.circle(frame, (cx, int(P_HEIGHT*self.height+i*PO_RB*self.width)+cx), int(cx-self.height*PT_BL), self.objects[i].bgr_color, -1)

#############################################################################
# functions for States
#############################################################################
    def show_analysis_graph(self, frame_time):
        image = np.zeros((self.height, self.width, 3), np.uint8)
        image[:,:] = (255, 255, 255)

        indent = self.height//30

        for j in range(len(self.objects)):
            obj = self.objects[j]
            art_obj = self.model_objects[j]
            last_detect = None
            
            cv2.line(image, (0, 6*indent), (self.width, 6*indent), (0, 0, 0), 8)

            for i in range(len(obj.change_times)):
                time = obj.change_times[i]

                # if we would draw outside of the window we end
                if indent*(time-frame_time) > self.height and indent*(art_obj.change_times[i]-1-frame_time) > self.height:
                    break
                
                if indent*(time-frame_time+6) < 0 and indent*(art_obj.change_times[i]-1-frame_time+6) < 0:
                    continue
                
                if i%2 == 1:
                    cv2.circle(image, (int(obj.catches[i//2]), int(indent*(time-frame_time+6))), obj.size, obj.bgr_color,  -1)
                    cv2.rectangle(image, (art_obj.path[art_obj.change_times[i]][0]-obj.size, int(indent*(art_obj.change_times[i]-frame_time+6))-obj.size), (art_obj.path[art_obj.change_times[i]][0]+obj.size, int(indent*(art_obj.change_times[i]-frame_time+6))+obj.size), obj.bgr_color, -1)
                else:
                    cv2.circle(image, (int(obj.throws[i//2]), int(indent*(time-frame_time+6))), obj.size, obj.bgr_color, 10)
                    cv2.rectangle(image, (art_obj.path[art_obj.change_times[i]][0]-obj.size, int(indent*(art_obj.change_times[i]-frame_time+6))-obj.size), (art_obj.path[art_obj.change_times[i]][0]+obj.size, int(indent*(art_obj.change_times[i]-frame_time+6))+obj.size), obj.bgr_color, 10)

            for i in range(-16, 34):
                if frame_time+i < len(obj.history) and frame_time+i > 0:
                    if obj.history[i+frame_time] != None:
                        if last_detect != None:
                            cv2.line(image, (obj.history[frame_time+last_detect][0], indent*(last_detect+6)), (obj.history[i+frame_time][0], indent*(i+6)), obj.bgr_color, 20)

                        last_detect = i

                cv2.line(image, (art_obj.path[(frame_time+i)%len(art_obj.path)][0], indent*(i+6)), (art_obj.path[(frame_time+i+1)%len(self.model_objects[j].path)][0], indent*(i+7)), (0, 0, 0), 20)

        return image

    def pattern_generator(self):
        catches = []
        throws = []
        flight_time = np.median(self.flight_time)
        dwell_time = np.median(self.dwell_time)

        for obj in self.objects:
            catches += obj.catches
            throws += obj.throws
            
        for i in range(len(self.objects)):
            self.model_objects.append(ModelBallObject(self.objects[i].size, self.objects[i].bgr_color))
            model_ball = self.model_objects[i]
            shift = 0
            j=0
            
            while len(model_ball.path) < len(self.objects[i].history)+flight_time+dwell_time:
                if len(model_ball.path)-int(2*i*(flight_time+dwell_time)//len(self.objects)) >= 0:
                    model_ball.change_times.append(len(model_ball.path)-int(2*i*(flight_time+dwell_time)//len(self.objects)))
                model_ball.generate_throw((int(np.median(throws[j%2::2])), self.catch_height), (int(np.median(catches[j%2::2])), self.catch_height), flight_time, np.median(self.gravity), shift)
                shift += flight_time
                shift -= int(shift)
                
                if len(model_ball.path)-int(2*i*(flight_time+dwell_time)//len(self.objects)) > 0 and len(model_ball.change_times) != 0:
                    model_ball.change_times.append(len(model_ball.path)-int(2*i*(flight_time+dwell_time)//len(self.objects)))
                model_ball.generate_hold((int(np.median(catches[j%2::2])), self.catch_height), (int(np.median(throws[(j+1)%2::2])), self.catch_height), dwell_time, self.catch_height+325, shift)
                shift += dwell_time
                shift -= int(shift)
                j+=1
                
            model_ball.path = model_ball.path[int(2*i*(flight_time+dwell_time)//len(self.objects)):] + model_ball.path[:int(2*i*(flight_time+dwell_time)//len(self.objects))]

    def detect_object(self, frame, hsv_frame):
        obj = self.objects[self.object_id]

        mask = cv2.inRange(hsv_frame, obj.hsv_m - obj.k_stdev * obj.hsv_stdev,
                           obj.hsv_m + obj.k_stdev * obj.hsv_stdev)

        mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ERO_SZ, ERO_SZ)))
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ERO_SZ, ERO_SZ)))

        if self.state == States.BALL_REFINEMENT:
            cv2.imshow(WNAME_MASK, cv2.resize(mask, (int(RESIZE*state.width), int(RESIZE*state.height))))

        cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(cnts) > 0:
            circle = cnts[0]
            
            for cnt in cnts:
                if abs(np.pi*cv2.minEnclosingCircle(circle)[1]**2-np.pi*obj.size**2) < abs(np.pi*cv2.minEnclosingCircle(cnt)[1]**2-np.pi*obj.size**2):
                    circle = cnt
                    
            ((x, y), radius) = cv2.minEnclosingCircle(circle)
            
            if radius > obj.size - C_SIZE_DEV*self.width and radius < obj.size + C_SIZE_DEV*self.width:
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
                cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), -1)

                return (int(x), int(y))
            
        return (None, None)

    def track_object(self, frame, hsv_frame):
        obj = self.objects[self.object_id]

        (x, y) = self.detect_object(frame, hsv_frame)
        
        if x != None:
            if self.catch_height < y and obj.in_air and len(obj.history) > 0:
                last = 1
                while obj.history[-last] == None:
                    last += 1
                    
                x_coor = []
                y_coor = []
                x_factor = []
                for l in range(len(obj.history[obj.last_throw:])):
                    if obj.history[obj.last_throw+l] != None:
                        x_coor.append(obj.history[obj.last_throw+l][0])
                        y_coor.append(obj.history[obj.last_throw+l][1])
                        if len(x_coor) > 2:
                            y_factor = (last_detect**2*(self.catch_height-y_coor[-1])-l**2*(self.catch_height-y_coor[-2]))/(l*last_detect*(last_detect-l))
                            x_factor.append((x_coor[-1]-x_coor[-2])/(l-last_detect))
                            self.gravity.append((2*y_factor*last_detect-2*(self.catch_height-y_coor[-2]))/last_detect**2)
                        last_detect = l

                x_factor = np.median(x_factor)

                if len(x_coor) > 3:
                    obj.parab, _ = scipy.optimize.curve_fit(parabola, x_coor, y_coor)
                    
                    intersections = solve_for_x(self.catch_height, *obj.parab)
                    if intersections != None:
                        if x_coor[0] > x_coor[2]:
                            obj.throws.append(max(intersections))
                            obj.catches.append(min(intersections))
                        else:
                            obj.catches.append(max(intersections))
                            obj.throws.append(min(intersections))

                    self.flight_time.append((obj.catches[-1]-obj.throws[-1])/x_factor)

                    if obj.last_catch != 0:
                        self.dwell_time.append(obj.last_throw + (obj.throws[-1]-obj.history[obj.last_throw][0])/x_factor - obj.last_catch)
                    obj.change_times.append(obj.last_throw + (obj.throws[-1]-obj.history[obj.last_throw][0])/x_factor)

                    obj.last_catch = len(obj.history)-last + (obj.catches[-1]-obj.history[-last][0])/x_factor
                    obj.change_times.append(obj.last_catch)
                obj.in_air = not obj.in_air

                    
            elif self.catch_height > y and not obj.in_air:
                obj.in_air = not obj.in_air
                obj.last_throw = len(obj.history)
                
            obj.history.append((x, y))
        
        else:
            obj.history.append(None)
        
    def analyze_objects(self, frame):
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv_frame = cv2.GaussianBlur(hsv_frame,(11,11),cv2.BORDER_DEFAULT)

        # track every ball
        for i in range(len(self.objects)):
            state.object_id = i
            state.track_object(frame, hsv_frame)

    def read_ball_color(self, frame):
        mask_in = self.get_ballmask()
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv_frame = cv2.GaussianBlur(hsv_frame,(11,11),cv2.BORDER_DEFAULT)

        hsv_m, hsv_stdev = cv2.meanStdDev(hsv_frame, mask = mask_in)

        mean = cv2.mean(frame, mask=mask_in)

        self.objects[self.object_id].set_params(mean[:3], hsv_m, hsv_stdev, self.circle_size)

#############################################################################
# frames for different states
#############################################################################
    def init_frame(self, frame):
        self.show_object_panel(frame)
        self.show_status(frame, "Initialization", f"{len(self.objects)} [SPACE], [A]dd object, [S]et height, [T]racking, [Q]uit")

    def tracking_frame(self, frame):
        self.show_status(frame, "Tracking", "[Q]uit")

    def analysis_frame(self, frame):
        self.show_status(frame,"Analysis", "[SPACE] Pause, [Q]uit")
        
        for ball in state.model_objects:
            cv2.circle(frame, ball.path[i%len(ball.path)], ball.size, ball.color, -1)

        for j in range(2*int(np.median(self.dwell_time) + np.median(self.flight_time) + 1)):
            cv2.line(frame, self.model_objects[0].path[j], self.model_objects[0].path[(j+1)%len(self.model_objects[0].path)], (0, 0, 255), 5) 

    def color_picker_frame(self, frame):
        self.show_status(frame, "Color picker", "[SPACE], [W][A][S][D], [Click], [T] <>, [G] ><, [ENTER], [Q]uit")
        cv2.circle(frame, (self.x, self.y), self.circle_size, C_COLOR, C_WIDTH)

    def ball_refinement_frame(self, frame):
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv_frame = cv2.GaussianBlur(hsv_frame,(11,11),cv2.BORDER_DEFAULT)
        self.detect_object(frame, hsv_frame)
        self.show_status(frame, "Ball refinement", "[SPACE], [C]olor, [ENTER], [Q]uit")

    def info_frame(self, frame):
        i = self.object_id
        self.show_status(frame, "Info", f"Object #{i+1}, [SPACE], [C]olor, [D]elete, [T]est, [ENTER], [Q]uit")
        cv2.circle(frame, self.center, int(C_CIRCLE*self.width), self.objects[i].bgr_color, -1)

    def set_height_frame(self, frame):
        cv2.line(frame, (0, self.catch_height), (self.width, self.catch_height), (0, 0, 0), 2)
        self.show_status(frame, "Set height", "[SPACE], [W] Up, [S] Down, [Click], [ENTER], [Q]uit")

#############################################################################
# General state showing function
#############################################################################
    def show_state(self, frame):
        if self.state == States.INIT:
            self.init_frame(frame)

        elif self.state == States.TRACKING:
            self.tracking_frame(frame)

        elif self.state == States.ANALYSIS:
            self.analysis_frame(frame)

        elif self.state == States.COLOR_PICKER:
            self.color_picker_frame(frame)

        elif self.state == States.BALL_REFINEMENT:
            self.ball_refinement_frame(frame)

        elif self.state == States.INFO:
            self.info_frame(frame)

        elif self.state == States.SET_HEIGHT:
            self.set_height_frame(frame)

#############################################################################
# Change state according the pressed key
#############################################################################
    def change_circle_size_by_key(self, key):
        if key == ord('t'):
            if self.circle_size < self.width/4:
                self.circle_size += 1
        elif key == ord('g'):
            if self.circle_size > 5:
                self.circle_size -= 1

    def change_by_key(self, key, frame):
#############################################################################
# INIT frame
        if self.state == States.INIT:
            if key == ord('t'):
                self.state = States.TRACKING
            elif ord('1') <= key <= ord('9'):
                n = key - ord('1')
                if n < len(self.objects):
                    self.object_id = n
                    self.state = States.INFO
            elif key == ord('a'):
                self.objects.append(BallObject(self.circle_size))
                self.object_id = len(self.objects) - 1
                self.state = States.COLOR_PICKER
            elif key == ord('p'):
                self.state = States.SNAPSHOT
            elif key == ord('r'):
                self.writer = cv2.VideoWriter(RECORD_FN,
                                              cv2.VideoWriter_fourcc(*'DIVX'),
                                              20, (self.width,self.height))
                self.rec_paused = False
                self.state = States.RECORDING
            elif key == ord('h'):
                self.state = States.SET_HEIGHT
            elif key == ord('q'):
                self.state = States.END

#############################################################################
# TRACKING frame
        elif self.state == States.TRACKING:
            if key == ord('q'): 
                self.state = States.END

#############################################################################
# ANALYSIS frame
        elif self.state == States.ANALYSIS:
            if key == 32: # SPACE
                while True:
                    key = cv2.waitKey(0)
                    if key == ord('q') or key == 32: # SPACE
                        break
            if key == ord('q'):
                self.state = States.END
                

#############################################################################
# COLOR PICKER frame
        elif self.state == States.COLOR_PICKER:
            cv2.setMouseCallback(WNAME_APP, self.get_ball_center_on_click)
            if key == 13: # ENTER
                self.read_ball_color(frame)
                print(f"Assigning params for object #{self.object_id}")
                self.state = States.BALL_REFINEMENT
            elif key == 27: # ESC
                self.objects.pop()
                self.object_id -= 1
                self.state = States.INIT
            elif key == ord('q'):
                self.state = States.END
            elif key == ord('w'):
                if self.y > self.circle_size:
                    self.y -= 1
            elif key == ord('s'):
                if self.y < self.height-self.circle_size:
                    self.y += 1
            elif key == ord('a'):
                if self.x > self.circle_size:
                    self.x -= 1
            elif key == ord('d'):
                if self.x < self.width-self.circle_size:
                    self.x += 1
            elif key == ord('t'):
                if self.circle_size < self.width/8:
                    self.circle_size += 1
            elif key == ord('g'):
                if self.circle_size > 5:
                    self.circle_size -= 1


#############################################################################
# BALL REFINEMENT frame
        elif self.state == States.BALL_REFINEMENT:
            if key == 13: # ENTER
                cv2.destroyWindow(WNAME_MASK)
                self.objects[self.object_id].history = []
                self.state = States.INFO
            elif key == ord('c'):
                self.state = States.COLOR_PICKER
            elif key == ord('q'):
                self.state = States.END


#############################################################################
# INFO frame
        elif self.state == States.INFO:
            if key == ord('d'):
                self.objects.pop(self.object_id)
                print(f"Deleting object #{self.object_id}")
                self.state = States.INIT
            elif key == 13: # ENTER
                self.state = States.INIT
            elif key == ord('c'):
                self.state = States.COLOR_PICKER
            elif key == ord('t'):
                self.state = States.BALL_REFINEMENT
            elif key == ord('q'):
                self.state = States.END


#############################################################################
# SET HEIGHT frame
        elif self.state == States.SET_HEIGHT:
            cv2.setMouseCallback(WNAME_APP, self.get_catch_height_on_click)
            if key == 13:
                self.state = States.INIT
            elif key == ord('w'):
                if self.catch_height > 0:
                    self.catch_height -= 1
            elif key == ord('s'):
                if self.catch_height < self.height:
                    self.catch_height += 1
            elif key == ord('q'):
                self.state = States.END

#############################################################################
# Application start
#############################################################################

filename = askopenfilename(filetypes=[
                    ("Video", ".mp4"),
                    ("Video", ".flv"),
                    ("Video", ".avi"),
                    ("Video", ".wmv"),
                    ("Video", ".mov"),
                    ("Video", ".mkv"),
                    ("Video", ".webm"),
                    ("Video", ".gif"),
                ])

if filename != "":
    cap = cv2.VideoCapture(filename)

    ret, frame = cap.read()
    state = AppState(frame)
    cv2.imshow(WNAME_APP, cv2.resize(frame, (int(RESIZE*state.width), int(RESIZE*state.height))))
    cv2.getWindowHandle()

    while cap.isOpened():
        frame_backup = frame.copy()
        
        if not ret:
            print("Can't receive frame. Exiting ...")

        while True and state.state != States.TRACKING:
            frame = frame_backup.copy()
            state.show_state(frame)
            cv2.imshow(WNAME_APP, cv2.resize(frame, (int(RESIZE*state.width), int(RESIZE*state.height))))
            key = cv2.waitKey(1)
            state.change_by_key(key, frame)

            if state.state == States.END or key == 32: # SPACE
                break

        if state.state == States.END or state.state == States.TRACKING:
            break
        
        ret, frame = cap.read()

    cv2.destroyAllWindows()

    cap = cv2.VideoCapture(filename)

    _, frame = cap.read()

    if state.state == States.TRACKING:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            state.show_state(frame)
            state.analyze_objects(frame)
            cv2.imshow(WNAME_APP, cv2.resize(frame, (int(RESIZE*state.width), int(RESIZE*state.height))))
            key = cv2.waitKey(1)
            state.change_by_key(key, frame)
            if state.state == States.END:
                exit()
            
        if state.state != States.END:
            cap.release()
            cv2.destroyAllWindows()

            state.pattern_generator()
            key = None

            # histogram of flight and dwell time
            fig = plt.figure()
            
            ax1 = plt.subplot(121)
            ax1.set_xlabel('Dwell time')
            ax1.axvline(x=np.median(state.dwell_time), color="y", linestyle="--")
            ax1.text(np.median(state.dwell_time), 0, "median", rotation="vertical", ha="right", va="bottom")
            ax1.hist(state.dwell_time, label="Dwell time")
            
            ax2 = plt.subplot(122)
            ax2.set_xlabel('Flight time')
            ax2.axvline(x=np.median(state.flight_time), color="y", linestyle="--")
            ax2.text(np.median(state.flight_time), 0, "median", rotation="vertical", ha="right", va="bottom")
            ax2.hist(state.flight_time, label="Flight time")
            
            fig.show()
            fig.canvas.flush_events()

            state.state = States.ANALYSIS

        while state.state != States.END:
            cap = cv2.VideoCapture(filename)

            i = 0
            while state.state != States.END:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                state.show_state(frame)
                cv2.imshow(WNAME_GRAPH, cv2.resize(state.show_analysis_graph(i), (int(RESIZE*state.width), int(RESIZE*state.height))))
                cv2.imshow(WNAME_APP, cv2.resize(frame, (int(RESIZE*state.width), int(RESIZE*state.height))))
                key = cv2.waitKey(30)
                state.change_by_key(key, frame)
                i+=1
        plt.close()
    cv2.destroyAllWindows()
    cap.release()





