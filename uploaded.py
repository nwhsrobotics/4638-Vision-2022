#!/usr/bin/env python3

# Copyright (c) FIRST and other WPILib contributors.
# Open Source Software; you can modify and/or share it under the terms of
# the WPILib BSD license file in the root directory of this project.

import json
import time
import sys
import cv2
import numpy
from bb_grip_contours import BlueBallGripPipeline
from rb_grip_contours import RedBallGripPipeline
from cscore import CameraServer, VideoSource, UsbCamera, MjpegServer, CvSink
from networktables import NetworkTablesInstance
from ReflectiveTapeContours import ReflectiveTapeContours


VIDEO_WIDTH = 320
VIDEO_HEIGHT = 240

#   JSON format:
#   {
#       "team": <team number>,
#       "ntmode": <"client" or "server", "client" if unspecified>
#       "cameras": [
#           {
#               "name": <camera name>
#               "path": <path, e.g. "/dev/video0">
#               "pixel format": <"MJPEG", "YUYV", etc>   // optional
#               "width": <video mode width>              // optional
#               "height": <video mode height>            // optional
#               "fps": <video mode fps>                  // optional
#               "brightness": <percentage brightness>    // optional
#               "white balance": <"auto", "hold", value> // optional
#               "exposure": <"auto", "hold", value>      // optional
#               "properties": [                          // optional
#                   {
#                       "name": <property name>
#                       "value": <property value>
#                   }
#               ],
#               "stream": {                              // optional
#                   "properties": [
#                       {
#                           "name": <stream property name>
#                           "value": <stream property value>
#                       }
#                   ]
#               }
#           }
#       ]
#       "switched cameras": [
#           {
#               "name": <virtual camera name>
#               "key": <network table key used for selection>
#               // if NT value is a string, it's treated as a name
#               // if NT value is a double, it's treated as an integer index
#           }
#       ]
#   }

configFile = "/boot/frc.json"

class CameraConfig: pass

team = None
server = False
cameraConfigs = []
switchedCameraConfigs = []
cameras = []

def parseError(str):
    """Report parse error."""
    print("config error in '" + configFile + "': " + str, file=sys.stderr)

def readCameraConfig(config):
    """Read single camera configuration."""
    cam = CameraConfig()

    # name
    try:
        cam.name = config["name"]
    except KeyError:
        parseError("could not read camera name")
        return False

    # path
    try:
        cam.path = config["path"]
    except KeyError:
        parseError("camera '{}': could not read path".format(cam.name))
        return False

    # stream properties
    cam.streamConfig = config.get("stream")

    cam.config = config

    cameraConfigs.append(cam)
    return True

def readSwitchedCameraConfig(config):
    """Read single switched camera configuration."""
    cam = CameraConfig()

    # name
    try:
        cam.name = config["name"]
    except KeyError:
        parseError("could not read switched camera name")
        return False

    # path
    try:
        cam.key = config["key"]
    except KeyError:
        parseError("switched camera '{}': could not read key".format(cam.name))
        return False

    switchedCameraConfigs.append(cam)
    return True

def readConfig():
    """Read configuration file."""
    global team
    global server

    # parse file
    try:
        with open(configFile, "rt", encoding="utf-8") as f:
            j = json.load(f)
    except OSError as err:
        print("could not open '{}': {}".format(configFile, err), file=sys.stderr)
        return False

    # top level must be an object
    if not isinstance(j, dict):
        parseError("must be JSON object")
        return False

    # team number
    try:
        team = j["team"]
    except KeyError:
        parseError("could not read team number")
        return False

    # ntmode (optional)
    if "ntmode" in j:
        str = j["ntmode"]
        if str.lower() == "client":
            server = False
        elif str.lower() == "server":
            server = True
        else:
            parseError("could not understand ntmode value '{}'".format(str))

    # cameras
    try:
        cameras = j["cameras"]
    except KeyError:
        parseError("could not read cameras")
        return False
    for camera in cameras:
        if not readCameraConfig(camera):
            return False

    # switched cameras
    if "switched cameras" in j:
        for camera in j["switched cameras"]:
            if not readSwitchedCameraConfig(camera):
                return False

    return True

def startCamera(config):
    """Start running the camera."""
    print("Starting camera '{}' on {}".format(config.name, config.path))
    inst = CameraServer.getInstance()
    camera = UsbCamera(config.name, config.path)
    server = inst.startAutomaticCapture(camera=camera, return_server=True)
    print(server)

    camera.setConfigJson(json.dumps(config.config))
    camera.setConnectionStrategy(VideoSource.ConnectionStrategy.kKeepOpen)

    if config.streamConfig is not None:
        server.setConfigJson(json.dumps(config.streamConfig))

    return camera

def startSwitchedCamera(config):
    """Start running the switched camera."""
    print("Starting switched camera '{}' on {}".format(config.name, config.key))
    server = CameraServer.getInstance().addSwitchedCamera(config.name)

    def listener(fromobj, key, value, isNew):
        if isinstance(value, float):
            i = int(value)
            if i >= 0 and i < len(cameras):
              server.setSource(cameras[i])
        elif isinstance(value, str):
            for i in range(len(cameraConfigs)):
                if value == cameraConfigs[i].name:
                    server.setSource(cameras[i])
                    break

    NetworkTablesInstance.getDefault().getEntry(config.key).addListener(
        listener,
        NetworkTablesInstance.NotifyFlags.IMMEDIATE |
        NetworkTablesInstance.NotifyFlags.NEW |
        NetworkTablesInstance.NotifyFlags.UPDATE)

    return server

def getExtrema(contours):
    contourList = contours
    contourPoints = contourList[0][:,0]
    sorted_contours = sorted(contourPoints, key=lambda tup: tup[0])

    min_point = sorted_contours[0]
    max_point = sorted_contours[len(sorted_contours)-1]


    return max_point, min_point

def runReflective(image, mainContours):
    found_contours = []
    num_found_countours = len(mainContours)
    avg_dist = 0
    avg_x_center_green = 0
    avg_y_center_green = 0
    for contours in mainContours:

        contourPoints = contours[:,0]

        x_points_green = contourPoints[:,0]
        y_points_green = contourPoints[:,1]

        x_min_green = numpy.amin(x_points_green)
        x_max_green = numpy.amax(x_points_green)
        green_width = x_max_green - x_min_green
        FOCAL_LENGTH = 374.8#289.1 #old 217.42   


        #call distance function to return widths
        Green_Real_Width = 5 #in
        perceived_distance = (FOCAL_LENGTH*Green_Real_Width)/green_width
        #print(green_width)

        y_min_green = numpy.amin(y_points_green)
        y_max_green = numpy.amax(y_points_green)

        x_center_green = ((x_max_green - x_min_green)/2) + x_min_green
        y_center_green = ((y_max_green - y_min_green)/2) + y_min_green

        


        #Draws center of balls
        #image = cv2.line(image, ((x_center_yellow).astype(numpy.int64),((y_center_yellow) - 15).astype(numpy.int64)),((x_center_yellow).astype(numpy.int64),((y_center_yellow) + 15).astype(numpy.int64)),(0,0,0),3)
        #image = cv2.line(image, (((x_center_yellow) - 15).astype(numpy.int64),(y_center_yellow).astype(numpy.int64)),(((x_center_yellow) + 15).astype(numpy.int64),(y_center_yellow).astype(numpy.int64)),(0,0,0),3)

        #Draws box around balls
        cv2.line(image, ((x_max_green).astype(numpy.int64),((y_max_green)).astype(numpy.int64)),((x_max_green).astype(numpy.int64),((y_min_green)).astype(numpy.int64)),(0,0,0),5)
        cv2.line(image, (((x_min_green)).astype(numpy.int64),(y_max_green).astype(numpy.int64)),(((x_min_green)).astype(numpy.int64),(y_min_green).astype(numpy.int64)),(0,0,0),5)
        cv2.line(image, ((x_max_green).astype(numpy.int64),((y_max_green)).astype(numpy.int64)),((x_min_green).astype(numpy.int64),((y_max_green)).astype(numpy.int64)),(0,0,0),5)
        cv2.line(image, (((x_max_green)).astype(numpy.int64),(y_min_green).astype(numpy.int64)),(((x_min_green)).astype(numpy.int64),(y_min_green).astype(numpy.int64)),(0,0,0),5)
        avg_dist += perceived_distance
        avg_x_center_green += x_center_green
        avg_y_center_green += y_center_green

        found_contours.append((perceived_distance, x_center_green, y_center_green, image)) #creating a tuple with all of the found contours
    
    

    avg_dist = avg_dist/num_found_countours
    avg_x_center_green = avg_x_center_green/num_found_countours
    avg_y_center_green = avg_y_center_green/num_found_countours
    cv2.circle(image, (int(avg_x_center_green), int(avg_y_center_green)), radius=7, color=(0, 255, 0), thickness=7)    
    
    return (avg_dist, avg_x_center_green, avg_y_center_green, image)





def runBall(image, mainContours, isRedAlliance):
    found_contours = []
    for contours in mainContours:

        contourPoints = contours[:,0]

        x_points_red = contourPoints[:,0]
        y_points_red = contourPoints[:,1]

        x_min_red = numpy.amin(x_points_red)
        x_max_red = numpy.amax(x_points_red)
        red_width = x_max_red - x_min_red
        FOCAL_LENGTH = 289.1 #old 217.42   
        



        #call distance function to return widths
        Red_Real_Width = 9.5 #in
        perceived_distance = (FOCAL_LENGTH*Red_Real_Width)/red_width


        y_min_red = numpy.amin(y_points_red)
        y_max_red = numpy.amax(y_points_red)

        x_center_red = ((x_max_red - x_min_red)/2) + x_min_red
        y_center_red = ((y_max_red - y_min_red)/2) + y_min_red

        


        #Draws center of balls
        #image = cv2.line(image, ((x_center_yellow).astype(numpy.int64),((y_center_yellow) - 15).astype(numpy.int64)),((x_center_yellow).astype(numpy.int64),((y_center_yellow) + 15).astype(numpy.int64)),(0,0,0),3)
        #image = cv2.line(image, (((x_center_yellow) - 15).astype(numpy.int64),(y_center_yellow).astype(numpy.int64)),(((x_center_yellow) + 15).astype(numpy.int64),(y_center_yellow).astype(numpy.int64)),(0,0,0),3)

        #Draws box around balls
        image = cv2.line(image, ((x_max_red).astype(numpy.int64),((y_max_red)).astype(numpy.int64)),((x_max_red).astype(numpy.int64),((y_min_red)).astype(numpy.int64)),(0,0,0),5)
        image = cv2.line(image, (((x_min_red)).astype(numpy.int64),(y_max_red).astype(numpy.int64)),(((x_min_red)).astype(numpy.int64),(y_min_red).astype(numpy.int64)),(0,0,0),5)
        image = cv2.line(image, ((x_max_red).astype(numpy.int64),((y_max_red)).astype(numpy.int64)),((x_min_red).astype(numpy.int64),((y_max_red)).astype(numpy.int64)),(0,0,0),5)
        image = cv2.line(image, (((x_max_red)).astype(numpy.int64),(y_min_red).astype(numpy.int64)),(((x_min_red)).astype(numpy.int64),(y_min_red).astype(numpy.int64)),(0,0,0),5)
    
        found_contours.append((perceived_distance, x_center_red, y_center_red, image)) #creating a tuple with all of the found contours
    
    closestBallData = tuple()
    shortestDistance = 10000000000000000
    '''
    This is a new filter'''
    filtered_contours = []
    for data_tup in found_contours:
        if (data_tup[2] > 90) and (data_tup[2] < 235):
            filtered_contours.append(data_tup)
    #end of new filter
    #for data_tup in found_contours:
    count = 0
    for data_tup in filtered_contours:
        if data_tup[0] < shortestDistance:
            shortestDistance = data_tup[0]
            closestBallData = data_tup
        count += 1
    #TODO: Add a safety feature here in case there is no data in the tuple, we wont draw the circle
    
    if (count > 0):
        if (isRedAlliance):
            cv2.circle(image, (int(closestBallData[1]), int(closestBallData[2])), radius=7, color=(0, 0, 255), thickness=7)    
        else: 
            cv2.circle(image, (int(closestBallData[1]), int(closestBallData[2])), radius=7, color=(255, 0, 0), thickness=7)    

        return closestBallData
    
    return -1, -1, -1, -1

def placeLine(pos, image):
    #line_divisor = sd.getNumber("Speed Constant", (5000/VIDEO_HEIGHT))
    #y_val = velocity/line_divisor
    y_val = int(VIDEO_HEIGHT - pos)

    cv2.line(image, (0, y_val), (VIDEO_WIDTH, y_val), (23, 177, 251), 2)


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        configFile = sys.argv[1]

    # read configuration
    if not readConfig():
        sys.exit(1)

    # start NetworkTables
    ntinst = NetworkTablesInstance.getDefault()
    if server:
        print("Setting up NetworkTables server")
        ntinst.startServer()
    else:
        print("Setting up NetworkTables client for team {}".format(team))
        ntinst.startClientTeam(team)
        ntinst.startDSClient()

    # start cameras
    for config in cameraConfigs:
        cameras.append(startCamera(config)) 

    # start switched cameras
    for config in switchedCameraConfigs:
        startSwitchedCamera(config)

    print("Camera Default Configurations Complete")


    RedGrip = RedBallGripPipeline()
    GreenGrip = ReflectiveTapeContours()
    BlueGrip = BlueBallGripPipeline()

    sinkA = CvSink("main cam")  
    sinkB = CvSink("reverse cam")

    sinkA.setSource(cameras[0]) #CAMERA ID
    sinkB.setSource(cameras[1])


    image_A = numpy.ndarray((VIDEO_WIDTH,VIDEO_HEIGHT,3), dtype = numpy.uint8)
    image_B = numpy.ndarray((VIDEO_WIDTH,VIDEO_HEIGHT,3), dtype = numpy.uint8)
    
    camservInst = CameraServer.getInstance()
    dashSource1 = camservInst.putVideo("UI Active Cam", VIDEO_WIDTH, VIDEO_HEIGHT) #creating a single main camera object

    sd = ntinst.getTable('SmartDashboard') #getting the smart dashboard object
    
    
    
    print("initalize complete")

    
    while True:
        isRedAlliance = sd.getBoolean("isRedAlliance", True)
        isReversed = sd.getBoolean("isReversed", False)
        timestamp,image_A = sinkA.grabFrame(image_A) #collecting the frame 
        timestamp, image_B = sinkB.grabFrame(image_B)
        GreenGrip.process(image_B)
        green_contours = GreenGrip.filter_contours_output
        
        
        if (isRedAlliance):
            RedGrip.process(image_A) #passing image_A and searching for the red ball
            red_contours = RedGrip.filter_contours_output
            main_contours = red_contours
        else:
            BlueGrip.process(image_A)
            blue_contours = BlueGrip.filter_contours_output
            main_contours = blue_contours
        
        motor_velocity = sd.getNumber("Motor Velocity", 0) #getting the motor velocity
        
        for contour in main_contours:
            cv2.drawContours(image_A, contour, -1, (0, 255, 0), 3)

        
        
        for contour in green_contours:
            cv2.drawContours(image_B, contour, -1, (0, 255, 0), 3)


        ball_dist = -1
        green_dist = -1
        x_center_ball = -1
        y_center_ball = -1
        if main_contours != []:
            ball_dist, x_center_ball, y_center_ball, image_A = runBall(image_A, main_contours, isRedAlliance)

            if (not x_center_ball == -1):
                x_center_ball = x_center_ball/VIDEO_WIDTH
                y_center_ball = y_center_ball/VIDEO_HEIGHT
                
        sd.putNumber('Ball X', x_center_ball)
        sd.putNumber('Ball Y', y_center_ball)
        sd.putNumber('Ball Distance', ball_dist)
        
        if (x_center_ball == -1):
            timestamp, image_A = sinkA.grabFrame(image_A) #get the frame again if there is nothing

        if green_contours != []:
            green_dist, x_center_green, y_center_green, image_B = runReflective(image_B, green_contours)

            #x center and y center is in terms of pixels, converting pixels to a value between 0 and 1
            x_center_green = x_center_green/VIDEO_WIDTH
            y_center_green = y_center_green/VIDEO_HEIGHT
            sd.putNumber('Green X', x_center_green)
            sd.putNumber('Green Y', y_center_green)
            sd.putNumber('Green Distance', green_dist)
        
        placeLine(VIDEO_HEIGHT-48, image_B)
        
        if (isReversed):
            dashSource1.putFrame(image_B)
        else:
            dashSource1.putFrame(image_A) #putting the postProcessed frame onto smartdashboard
        
        
        #TODO: Make sure to publish the contours report onto SmartDashboard
       


        

        
'''
@sd Usage
* sd.getNumber("KEY_NAME", defaultValue)
* sd.putNumber('KEY_NAME', value)

code for approximating the circularity
#approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)
            #len(approx) -- this returns the circularity approximation

'''