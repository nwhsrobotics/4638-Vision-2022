#!/usr/bin/env python3
#----------------------------------------------------------------------------
# Copyright (c) 2018 FIRST. All Rights Reserved.
# Open Source Software - may be modified and shared by FRC teams. The code
# must be accompanied by the FIRST BSD license file in the root directory of
# the project.
#----------------------------------------------------------------------------

import json
import time
import sys

from cscore import CameraServer, VideoSource, UsbCamera, MjpegServer, CvSink
from networktables import NetworkTablesInstance
import cv2
import numpy
import math


from reflective_tape_new import GripPipelineGreen
from yellow_ball_test import GripPipelineYellow


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

    #TESTING TO SEE IF CAMERAS TURN ON WHEN UNCOMMETED

    server = inst.startAutomaticCapture(camera=camera, return_server=True) #RECOMMENT

    camera.setConfigJson(json.dumps(config.config))
    camera.setConnectionStrategy(VideoSource.ConnectionStrategy.kKeepOpen) #RECOMMENT

    if config.streamConfig is not None: #RECOMMENT
        server.setConfigJson(json.dumps(config.streamConfig)) #RECOMMENT

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

#Distance function

def distance_to_camera_yellow(Width, perceivedWidth):

    #Find focal distance
    Control_Distance = 25
    Control_Width_pixels = 57
    Control_Width_in = 7
    focalLength = (Control_Width_pixels * Control_Distance) / Control_Width_in

    return (Width * focalLength) / perceivedWidth

def distance_to_camera_green(Width, perceivedWidth):

    #Find focal distance
    Control_Distance = 120
    Control_Width_pixels = 79
    Control_Width_in = 44
    focalLength = (Control_Width_pixels * Control_Distance) / Control_Width_in

    return (Width * focalLength) / perceivedWidth

def angleFinderG():

    max_point, min_point = getContourG();
    slope = (min_point[1] - max_point[1])/(min_point[0]-max_point[0])
    Width = max_point[0] - min_point[0]

    control_angle = 45 #deg
    control_slope =.36
    angle = (control_angle * slope)/control_slope

    sd.putNumber('Green Angle', angle)
    sd.putNumber('Green Width', Width)

    return angle, Width

def getContourG():
    contourList = contours_output_green
    contourPoints = contourList[0][:,0]
    sorted_contours = sorted(contourPoints, key=lambda tup: tup[0])

    min_point = sorted_contours[0]
    max_point = sorted_contours[len(sorted_contours)-1]


    return max_point, min_point

def centerPointsG():
    #Get closest and farthest x
    max_point, min_point = getContourG()

    x_center = ((max_point[0]-min_point[0])/2) + min_point[0]
    if (max_point[1] > min_point[1]):
        y_center = ((max_point[1] - min_point[1])/2) + min_point[1]

    elif (max_point[1] < min_point[1]):
        y_center = ((min_point[1] - max_point[1])/2) + max_point[1]

    else:
        y_center = min_point[1]

    sd.putNumber('Center X Green', x_center)
    sd.putNumber('Center Y Green', y_center)

    center_Points = [x_center, y_center]
    return center_Points

def getValuesGreen(image):

    #get angle of object, get the width of object, min and max listsss
    angle_green, Green_Width = angleFinderG()
    center_Points = centerPointsG()
    max_point, min_point = getContourG()

    #call distance function to return widths
    Green_Real_Width = 44 #in
    inchesG = distance_to_camera_green(Green_Real_Width, Green_Width) - (0.0111*(angle_green*angle_green)) + (0.0809*angle_green) #quadratic formula approximates real distance
    sd.putNumber('Green Distance', inchesG)
    
    #draw crosshair
    image = cv2.line(image, ((center_Points[0]).astype(numpy.int64), (center_Points[1] - 50).astype(numpy.int64)), ((center_Points[0]).astype(numpy.int64), (center_Points[1] +50).astype(numpy.int64)), (255,0,255), 2)

    #Draw line from corner to corner
    image = cv2.line(image, (max_point[0],max_point[1]),(min_point[0],min_point[1]),(255,0,255),3)

    image = cv2.putText(image, "Distance={}in".format(inchesG.astype(numpy.int64)),((center_Points[0] - 70).astype(numpy.int64), (center_Points[1] +70).astype(numpy.int64)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 2)

    return image

def getValuesYellow(image):

    inchesZ = 10000

    for contours in contours_output_yellow:

        contourPoints = contours[:,0]

        x_points_yellow = contourPoints[:,0]
        y_points_yellow = contourPoints[:,1]

        x_min_yellow = numpy.amin(x_points_yellow)
        x_max_yellow = numpy.amax(x_points_yellow)

        #for distance
        Yellow_Width = x_max_yellow - x_min_yellow
        #print(Yellow_Width)

        #call distance function to return widths
        Yellow_Real_Width = 7 #in
        inchesY = distance_to_camera_yellow(Yellow_Real_Width, Yellow_Width)

        y_min_yellow = numpy.amin(y_points_yellow)
        y_max_yellow = numpy.amax(y_points_yellow)

        x_center_yellow = ((x_max_yellow - x_min_yellow)/2) + x_min_yellow
        y_center_yellow = ((y_max_yellow - y_min_yellow)/2) + y_min_yellow

        #Draws center of balls
        #image = cv2.line(image, ((x_center_yellow).astype(numpy.int64),((y_center_yellow) - 15).astype(numpy.int64)),((x_center_yellow).astype(numpy.int64),((y_center_yellow) + 15).astype(numpy.int64)),(0,0,0),3)
        #image = cv2.line(image, (((x_center_yellow) - 15).astype(numpy.int64),(y_center_yellow).astype(numpy.int64)),(((x_center_yellow) + 15).astype(numpy.int64),(y_center_yellow).astype(numpy.int64)),(0,0,0),3)

        #Draws box around balls
        image = cv2.line(image, ((x_max_yellow).astype(numpy.int64),((y_max_yellow)).astype(numpy.int64)),((x_max_yellow).astype(numpy.int64),((y_min_yellow)).astype(numpy.int64)),(0,0,0),1)
        image = cv2.line(image, (((x_min_yellow)).astype(numpy.int64),(y_max_yellow).astype(numpy.int64)),(((x_min_yellow)).astype(numpy.int64),(y_min_yellow).astype(numpy.int64)),(0,0,0),1)
        image = cv2.line(image, ((x_max_yellow).astype(numpy.int64),((y_max_yellow)).astype(numpy.int64)),((x_min_yellow).astype(numpy.int64),((y_max_yellow)).astype(numpy.int64)),(0,0,0),1)
        image = cv2.line(image, (((x_max_yellow)).astype(numpy.int64),(y_min_yellow).astype(numpy.int64)),(((x_min_yellow)).astype(numpy.int64),(y_min_yellow).astype(numpy.int64)),(0,0,0),1)

        #Display Distance
        image = cv2.putText(image, "Distance={}in".format(inchesY.astype(numpy.int64)),((x_center_yellow - 40).astype(numpy.int64), (y_center_yellow + 40).astype(numpy.int64)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        if (inchesY < inchesZ):
            sd.putNumber('Center X Yellow', x_center_yellow)
            sd.putNumber('Center Y Yellow', y_center_yellow)

            sd.putNumber('Yellow Distance', inchesY)

            inchesZ = inchesY

    return image





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

    # start cameras
    for config in cameraConfigs:
        cameras.append(startCamera(config))

    # start switched cameras
    for config in switchedCameraConfigs:
        startSwitchedCamera(config)

    #Init Grip
    grip_green = GripPipelineGreen()
    grip_yellow = GripPipelineYellow()
    sinkY = CvSink("vision Yellow")
    sinkF = CvSink("vision Front")
    sinkY.setSource(cameras[1]) #Was 0
    sinkF.setSource(cameras[2]) #Was 1, trying 0
    sinkG = CvSink("vision Green")
    sinkG.setSource(cameras[0]) #Was 1
    image_G = numpy.ndarray((320,240,3), dtype = numpy.uint8) #Mid val was 360
    image_F = numpy.ndarray((320,240,3), dtype = numpy.uint8) #Mid val was 360
    image_Y = numpy.ndarray((320,240,3), dtype = numpy.uint8) #Mid val was 360
    camservInst = CameraServer.getInstance()

    dashSource1 = camservInst.putVideo("UI Active Cam", 320, 240)
    #dashSource2 = camservInst.putVideo("UI Green Cam", 320, 240)
    # loop forever

    sd = ntinst.getTable('SmartDashboard')

    while True:

        #Change the time front camera
        timestamp,image_F = sinkF.grabFrame(image_F)

        #change time yellow ball camera
        timestamp,image_Y = sinkY.grabFrame(image_Y)
        grip_yellow.process(image_Y)

        #change time shooter camera
        timestamp,image_G = sinkG.grabFrame(image_G)
        grip_green.process(image_G)


        contours_output_green = grip_green.filter_contours_output
        contours_output_yellow = grip_yellow.filter_contours_output

        if(contours_output_green):
            image_G = getValuesGreen(image_G)

        #We need to reset smartdashboard values
        else :
            angle = -1
            sd.putNumber('Green Angle', angle)

            x_center = -1
            y_center = -1
            sd.putNumber('Center X Green', x_center)
            sd.putNumber('Center Y Green', y_center)

            #only get yellow distance
            inchesG = -1
            sd.putNumber('Green Distance', inchesG)

        if(contours_output_yellow):
            image_Y = getValuesYellow(image_Y)

    #We need to reset smartdashboard values
        else:
            x_center_yellow = -1
            y_center_yellow = -1

            sd.putNumber('Center X Yellow', x_center_yellow)
            sd.putNumber('Center Y Yellow', y_center_yellow)

            #only get green distance
            inchesY = -1
            sd.putNumber('Yellow Distance', inchesY)

        #END OF IF STATEMENTS/SEND DATA


        camera_chooser = sd.getNumber("Camera chooser", 1) ## Starts on back camera to pick up balls

        if (camera_chooser == 1):
            dashSource1.putFrame(image_Y)
        elif (camera_chooser == 2):
            dashSource1.putFrame(image_F)
        elif (camera_chooser == 3):
            dashSource1.putFrame(image_G)




#
# def getValuesBoth(image):
#
#     contourPoints_green = contours_output_green[0][:,0]
#
#     x_points_green = contourPoints_green[:,0]
#     y_points_green = contourPoints_green[:,1]
#
#     x_min_green = numpy.amin(x_points_green)
#     x_max_green = numpy.amax(x_points_green)
#
#     sorted_contours = sorted(contourPoints_green, key=lambda tup: tup[0])
#     #print(sorted_contours_min[0])
#     min_point = sorted_contours[0]
#     #print(min_point)
#     max_point = sorted_contours[len(sorted_contours)-1]
#     #print(max_point)
#
#     sorted_contours = sorted(contourPoints_green, key=lambda tup: tup[0])
#     #print(sorted_contours_min[0])
#     min_point = sorted_contours[0]
#     #print(min_point)
#     max_point = sorted_contours[len(sorted_contours)-1]
#     #print(max_point)
#
#     y_min_green = numpy.amin(y_points_green)
#     y_max_green = numpy.amax(y_points_green)
#     #for distance
#     Green_Width = x_max_green - x_min_green
#     #sd.putNumber('Green Width', Green_Width)
#
#     #call distance function to return widths
#     Green_Real_Width = 39 #in
#
#     #Find slope of the target
#     slope = (min_point[1] - max_point[1])/(min_point[0]-max_point[0])
#
#     #Find angle of the target
#     angle_green = angleFinderG(slope)
#
#     #find distance
#     inchesG = distance_to_camera(Green_Real_Width, Green_Width) - (0.0111*(angle_green*angle_green)) + (0.0809*angle_green)
#     sd.putNumber('Green Distance', inchesG)
#
#     #sd.putNumber('Min X Green', x_min_green)
#     #sd.putNumber('Max X Green', x_max_green)
#     #sd.putNumber('Min Y Green', y_min_green)
#     #sd.putNumber('Max Y Green', y_max_green)
#
#
#     area_green = (x_max_green - x_min_green) * (y_max_green - y_min_green)
#
#     #sd.putNumber('Green Area', area_green)
#
#     x_center_green = ((max_point[0]-min_point[0])/2) + min_point[0]
#     if (max_point[1] > min_point[1]):
#         y_center_green = ((max_point[1] - min_point[1])/2) + min_point[1]
#
#     elif (max_point[1] < min_point[1]):
#         y_center_green = ((min_point[1] - max_point[1])/2) + max_point[1]
#
#     else:
#         y_center_green = min_point[1]
#
#     sd.putNumber('Center X Green', x_center_green)
#     sd.putNumber('Center Y Green', y_center_green)
#
#     #image = cv2.line(image, ((x_center_green).astype(numpy.int64),((y_center_green) - 15).astype(numpy.int64)),((x_center_green).astype(numpy.int64),((y_center_green) + 15).astype(numpy.int64)),(255,0,255),5)
#     #image = cv2.line(image, (((x_center_green) - 15).astype(numpy.int64),(y_center_green).astype(numpy.int64)),(((x_center_green) + 15).astype(numpy.int64),(y_center_green).astype(numpy.int64)),(255,0,255),5)
#
#     #image = cv2.line(image, ((x_max_green).astype(numpy.int64),((y_max_green)).astype(numpy.int64)),((x_max_green).astype(numpy.int64),((y_min_green)).astype(numpy.int64)),(255,0,255),5)
#     #image = cv2.line(image, (((x_min_green)).astype(numpy.int64),(y_max_green).astype(numpy.int64)),(((x_min_green)).astype(numpy.int64),(y_min_green).astype(numpy.int64)),(255,0,255),5)
#     #image = cv2.line(image, ((x_max_green).astype(numpy.int64),((y_max_green)).astype(numpy.int64)),((x_min_green).astype(numpy.int64),((y_max_green)).astype(numpy.int64)),(255,0,255),5)
#     #image = cv2.line(image, (((x_max_green)).astype(numpy.int64),(y_min_green).astype(numpy.int64)),(((x_min_green)).astype(numpy.int64),(y_min_green).astype(numpy.int64)),(255,0,255),5)
#
#     #Display Distance
#     image = cv2.putText(image, "Distance={}in".format(inchesG.astype(numpy.int64)),((x_center_green - 50).astype(numpy.int64), (y_center_green +50).astype(numpy.int64)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 3)
#
#     image = cv2.circle(image, (max_point[0], max_point[1]), 5, (0,0,255), -1)
#     image = cv2.circle(image, (min_point[0], min_point[1]), 5, (0,0,255), -1)
#
#     image = cv2.line(image, ((max_point[0]),(max_point[1])),(min_point[0],min_point[1]),(255,0,255),5)
#
#     #Draw Crosshair
#     image = cv2.line(image, ((x_center_green).astype(numpy.int64), (y_center_green - 50).astype(numpy.int64)), ((x_center_green).astype(numpy.int64), (y_center_green +50).astype(numpy.int64)), (255,0,255), 3)
#
#     #Angle
#     image = cv2.putText(image, "Angle={}deg".format(angle_green.astype(numpy.int64)),((x_center_green - 50).astype(numpy.int64), (y_center_green +90).astype(numpy.int64)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 3)
#
#     #Start of Yellow Code
#     inchesZ = 10000
#
#     for contours in contours_output_yellow:
#
#         contourPoints = contours[:,0]
#
#         x_points_yellow = contourPoints[:,0]
#         y_points_yellow = contourPoints[:,1]
#
#         x_min_yellow = numpy.amin(x_points_yellow)
#         x_max_yellow = numpy.amax(x_points_yellow)
#
#         #for distance
#         Yellow_Width = x_max_yellow - x_min_yellow
#         #sd.putNumber('Yellow Width', Yellow_Width)
#
#         #call distance function to return widths
#         Yellow_Real_Width = 7 #in
#         inchesY = distance_to_camera(Yellow_Real_Width, Yellow_Width)
#         #sd.putNumber('Yellow Distance', inchesY)
#
#         #sd.putNumber('YellowDistance', inchesY)
#
#         y_min_yellow = numpy.amin(y_points_yellow)
#         y_max_yellow = numpy.amax(y_points_yellow)
#
#         #sd.putNumber('Min X Yellow', x_min_yellow)
#         #sd.putNumber('Max X Yellow', x_max_yellow)
#         #sd.putNumber('Min Y Yellow', y_min_yellow)
#         #sd.putNumber('Max Y Yellow', y_max_yellow)
#
#         area_yellow = (x_max_yellow - x_min_yellow) * (y_max_yellow - y_min_yellow)
#
#         #sd.putNumber('Yellow Area', area_yellow)
#
#         x_center_yellow = ((x_max_yellow - x_min_yellow)/2) + x_min_yellow
#         y_center_yellow = ((y_max_yellow - y_min_yellow)/2) + y_min_yellow
#
#         #sd.putNumber('Center X Yellow', x_center_yellow)
#         #sd.putNumber('Center Y Yellow', y_center_yellow)
#
#         image = cv2.line(image, ((x_center_yellow).astype(numpy.int64),((y_center_yellow) - 15).astype(numpy.int64)),((x_center_yellow).astype(numpy.int64),((y_center_yellow) + 15).astype(numpy.int64)),(0,0,0),5)
#         image = cv2.line(image, (((x_center_yellow) - 15).astype(numpy.int64),(y_center_yellow).astype(numpy.int64)),(((x_center_yellow) + 15).astype(numpy.int64),(y_center_yellow).astype(numpy.int64)),(0,0,0),5)
#
#         image = cv2.line(image, ((x_max_yellow).astype(numpy.int64),((y_max_yellow)).astype(numpy.int64)),((x_max_yellow).astype(numpy.int64),((y_min_yellow)).astype(numpy.int64)),(0,0,0),5)
#         image = cv2.line(image, (((x_min_yellow)).astype(numpy.int64),(y_max_yellow).astype(numpy.int64)),(((x_min_yellow)).astype(numpy.int64),(y_min_yellow).astype(numpy.int64)),(0,0,0),5)
#         image = cv2.line(image, ((x_max_yellow).astype(numpy.int64),((y_max_yellow)).astype(numpy.int64)),((x_min_yellow).astype(numpy.int64),((y_max_yellow)).astype(numpy.int64)),(0,0,0),5)
#         image = cv2.line(image, (((x_max_yellow)).astype(numpy.int64),(y_min_yellow).astype(numpy.int64)),(((x_min_yellow)).astype(numpy.int64),(y_min_yellow).astype(numpy.int64)),(0,0,0),5)
#
#         #Display Distance
#         image = cv2.putText(image, "Distance={}in".format(inchesY.astype(numpy.int64)),((x_center_yellow - 70).astype(numpy.int64), (y_center_yellow +70).astype(numpy.int64)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)
#
#         if (inchesY < inchesZ):
#             sd.putNumber('Center X Yellow', x_center_yellow)
#             sd.putNumber('Center Y Yellow', y_center_yellow)
#             #sd.putNumber('Yellow Area', area_yellow)
#
#             #sd.putNumber('Min X Yellow', x_min_yellow)
#             #sd.putNumber('Max X Yellow', x_max_yellow)
#             #sd.putNumber('Min Y Yellow', y_min_yellow)
#             #sd.putNumber('Max Y Yellow', y_max_yellow)
#
#             sd.putNumber('Yellow Distance', inchesY)
#
#
#             #sd.putNumber('Yellow Width', Yellow_Width)
#
#             inchesZ = inchesY
#
#     return image
