#!/usr/bin/env python3

# Copyright (c) FIRST and other WPILib contributors.
# Open Source Software; you can modify and/or share it under the terms of
# the WPILib BSD license file in the root directory of this project.

import json
import time
import sys
import cv2
import numpy
from rb_grip_contours import RedBallGripContours
from cscore import CameraServer, VideoSource, UsbCamera, MjpegServer, CvSink
from networktables import NetworkTablesInstance

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

def runBall(image, mainContours):
    for contours in mainContours:

        contourPoints = contours[:,0]

        x_points_red = contourPoints[:,0]
        y_points_red = contourPoints[:,1]

        x_min_red = numpy.amin(x_points_red)
        x_max_red = numpy.amax(x_points_red)
        red_width = x_max_red - x_min_red
        FOCAL_LENGTH = 289.1 #old 217.42   

        #for distance
        Red_Width = x_max_red - x_min_red   


        #call distance function to return widths
        Red_Real_Width = 9.5 #in
        perceived_distance = (FOCAL_LENGTH*Red_Real_Width)/red_width


        y_min_red = numpy.amin(y_points_red)
        y_max_red = numpy.amax(y_points_red)

        x_center_yellow = ((x_max_red - x_min_red)/2) + x_min_red
        y_center_yellow = ((y_max_red - y_min_red)/2) + y_min_red
        #print(f"Distance: {perceivedDistance}")

        #Draws center of balls
        #image = cv2.line(image, ((x_center_yellow).astype(numpy.int64),((y_center_yellow) - 15).astype(numpy.int64)),((x_center_yellow).astype(numpy.int64),((y_center_yellow) + 15).astype(numpy.int64)),(0,0,0),3)
        #image = cv2.line(image, (((x_center_yellow) - 15).astype(numpy.int64),(y_center_yellow).astype(numpy.int64)),(((x_center_yellow) + 15).astype(numpy.int64),(y_center_yellow).astype(numpy.int64)),(0,0,0),3)

        #Draws box around balls
        image = cv2.line(image, ((x_max_red).astype(numpy.int64),((y_max_red)).astype(numpy.int64)),((x_max_red).astype(numpy.int64),((y_min_red)).astype(numpy.int64)),(0,0,0),5)
        image = cv2.line(image, (((x_min_red)).astype(numpy.int64),(y_max_red).astype(numpy.int64)),(((x_min_red)).astype(numpy.int64),(y_min_red).astype(numpy.int64)),(0,0,0),5)
        image = cv2.line(image, ((x_max_red).astype(numpy.int64),((y_max_red)).astype(numpy.int64)),((x_min_red).astype(numpy.int64),((y_max_red)).astype(numpy.int64)),(0,0,0),5)
        image = cv2.line(image, (((x_max_red)).astype(numpy.int64),(y_min_red).astype(numpy.int64)),(((x_min_red)).astype(numpy.int64),(y_min_red).astype(numpy.int64)),(0,0,0),5)
    
    return perceived_distance, image
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


    

    RedGrip = RedBallGripContours()

    sinkA = CvSink("main cam")  

    sinkA.setSource(cameras[1])

    image_A = numpy.ndarray((320,240,3), dtype = numpy.uint8)

    
    camservInst = CameraServer.getInstance()
    dashSource1 = camservInst.putVideo("UI Active Cam", 320, 240) #creating a single main camera object

    sd = ntinst.getTable('SmartDashboard') #getting the smart dashboard object
    
    print("initalize complete")

    
    while True:
        timestamp,image_A = sinkA.grabFrame(image_A) #collecting the frame 
        RedGrip.process(image_A) #passing image_A and searching for the red ball
        #image_A = cv2.drawKeypoints(image_A, RedGrip.find_blobs_output, outputImage = None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) #drawing out the keypoints onto the image
        contours = RedGrip.filter_contours_output
        for contour in contours:
            cv2.drawContours(image_A, contour, -1, (0, 255, 0), 3)
        red_dist = -1
        if contours != []:
            red_dist, image_A = runBall(image_A, contours)
        dashSource1.putFrame(image_A) #putting the postProcessed frame onto smartdashboard
        sd.putNumber('Red Ball Distance', red_dist)
        #TODO: Make sure to publish the contours report onto SmartDashboard
       

        

        
'''
@sd Usage
* sd.getNumber("KEY_NAME", defaultValue)
* sd.putNumber('KEY_NAME', value)

'''