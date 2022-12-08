import cv2
import numpy as np
from PIL import Image
# from vpython import *
import math

# arduino
import time
import serial


class Arduino:
    def __init__(self, comPort, baudRate, setupTimeDelay = 1):
        self.comPort = comPort
        self.baudRate = baudRate
        self.arduinoData = serial.Serial(self.comPort, self.baudRate)
        self.currentVPos = (0,0)

        if (setupTimeDelay > 0):
            time.sleep(setupTimeDelay)
        return

    def readFromArduino(self):
        if (self.arduinoData.inWaiting() == 0):
            return None
        else:
            return self.arduinoData.readline()

    def readUntilData(self, cleanData = True):
        while (self.arduinoData.inWaiting() == 0):
            pass
        dataPacket = self.arduinoData.readline()
        if (cleanData):
            dataPacket = str(dataPacket,"utf-8")
            dataPacket = dataPacket.strip("\r\n")
        return dataPacket

    def writeToArduino(self, dataPacket):
        if (len(dataPacket) < 2 or dataPacket[-2] != '\\' or dataPacket[-1] != 'r'):
            dataPacket += "\r"
        self.arduinoData.write(dataPacket.encode())

    def writeWaitForAnswer(self, dataPacket, cleanData = True):
        self.writeToArduino(dataPacket)
        return self.readUntilData(cleanData)

class Drawer:        
    def __init__(self, arduino, realCanvasDims, motorStepsPerRev, motorWheelRadius, pixelsPerCM = None):
        self.arduino = arduino
        self.fullCanvas = realCanvasDims
        self.correctedCanvas = realCanvasDims
        self.motorStepsPerRev = motorStepsPerRev
        self.motorWheelRad = motorWheelRadius
        self.pixelsPerCM = pixelsPerCM
        self.virtualDims = None
        self.offset = (0,0)
        self.outputVals = False
        return

    def recalcParams(self, imgDims, maintainAspec = True):
        width = self.fullCanvas[0]
        height = self.fullCanvas[1]
        imgWidth = imgDims[0]
        imgHeight = imgDims[1]
        self.virtualDims = imgDims
        self.offset = (0,0)
        if (maintainAspec):    
            imgRatio = imgWidth/imgHeight
            canvasRatio = width/height
            if (imgRatio >= canvasRatio):
                self.correctedCanvas = (width, width/imgRatio)
                self.offset = (0,(height - self.correctedCanvas[1])/2)
            else:
                self.correctedCanvas = (height*imgRatio, height)
                self.offset = ((width - self.correctedCanvas[0])/2,0)
        if (self.pixelsPerCM is None):
            self.pixelsPerCM = int(self.virtualDims[0]//self.correctedCanvas[0]) if (self.correctedCanvas[0] <= self.correctedCanvas[1] or self.correctedCanvas[1] == 0) else int(self.virtualDims[1]//self.correctedCanvas[1])
        # For logging and debugging
        if (self.outputVals):
            print("recalcParams: ")
            print("    imgDims", imgDims)
            print("    fullCanvas", self.fullCanvas)
            print("    correctedCanvas", self.correctedCanvas)
            print("    offset", self.offset)

    def pixelsToSteps(self, pixelCoord):
        realCoord = (pixelCoord[0]/self.pixelsPerCM + self.offset[0], pixelCoord[1]/self.pixelsPerCM + self.offset[1])
        
        wheelCircumfrence = 2*math.pi*self.motorWheelRad
        fullRotations = (realCoord[0]//wheelCircumfrence, realCoord[1]//wheelCircumfrence)
        partialRotations = (realCoord[0]%wheelCircumfrence, realCoord[1]%wheelCircumfrence)
        
        fullRotationSteps = (fullRotations[0] * self.motorStepsPerRev, fullRotations[1] * self.motorStepsPerRev)
        partialRotationSteps = (partialRotations[0]/wheelCircumfrence * self.motorStepsPerRev, partialRotations[1]/wheelCircumfrence * self.motorStepsPerRev)

        steps = (fullRotationSteps[0] + partialRotationSteps[0], fullRotationSteps[1] + partialRotationSteps[1])

        steps = (round(steps[0]), round(steps[1]))
        
        # For logging and debugging
        if (self.outputVals):
            print("PixelsToSteps")
            print("---Pixel Coords = ", pixelCoord, "pixels")
            print("---Offset = ", self.offset, "pixels")
            print("---Real Coords = ", realCoord, "cm")
            print("-----pixelsPerCM = ", self.pixelsPerCM, "pixels/cm")
            print("-----Wheel Circ = ", wheelCircumfrence, "cm")
            print("---Full Rotations = ", fullRotations)
            print("-----Full Rotation Steps = ", fullRotationSteps)
            print("---Partial Rotations = ", partialRotations)
            print("-----Partial Rotation Steps = ", partialRotationSteps)
            print("---Total Steps = ", steps, "steps")
            print("----------------------------------------------------------")
            
        return steps

    def findMax(self, instructions):
        mX = 0
        mY = 0
        for line in instructions:
            for pixel in line:
                if (pixel[0] > mX):
                    mX = pixel[0]
                if (pixel[1] > mY):
                    mY = pixel[1]
        return (mX,mY)

    def drawIns(self, instructions):
        dims = self.findMax(instructions)
        self.recalcParams(dims)
        self.draw(instructions)
        return

    def getInstructions(self, image):
        if (self.pixelsPerCM is not None):
            instructions, newDims = ImageProcessor.getDrawInstructions(image = image, resizeDims = (self.fullCanvas[0]*self.pixelsPerCM, self.fullCanvas[1]*self.pixelsPerCM))
        else:
            instructions, newDims = ImageProcessor.getDrawInstructions(image = image)
        # For logging and debugging
        if (self.outputVals):
            print("newDims: ", newDims)
        self.recalcParams(newDims)
        return instructions

    def drawImg(self, image):
        instructions = self.getInstructions(image)
        self.draw(instructions)
        return


    def draw(self, instructions):
        log = ""
        log += time.strftime("%H:%M:%S", time.localtime()) + " : draw()\n"
        for line in instructions:
            self.moveTo(line[0])
            self.drawLine(line)
        self.moveTo((0,0))
        log += time.strftime("%H:%M:%S", time.localtime()) + " : ============================END=====================================\n"
        # For logging and debugging
        if (self.outputVals):
            print(log)
        return

    def drawLine(self, line):
        log = ""
        log += time.strftime("%H:%M:%S", time.localtime()) + " : drawLine(" + str(line) + ")\n"
        i = 0
        for vCoord in line:
            log += "    " + time.strftime("%H:%M:%S", time.localtime()) + " :[" + str(i) + "] vCoord(" + str(vCoord[0]) + "," + str(vCoord[1]) +")\n"
            steps = self.pixelsToSteps(vCoord)
            log += "    " + time.strftime("%H:%M:%S", time.localtime()) + " :[" + str(i) + "] writeToArduino(" + str(steps[0]) + "," + str(steps[1]) + ",DRAW)\n"
            # self.arduino.writeToArduino(str(steps[0]) + "," + str(steps[1]) + "," + str(0))
            self.arduino.writeToArduino(str(steps[0]) + "," + str(steps[1]) + "," + str(1))
            dataBack = self.arduino.readUntilData()
            log += "    " + time.strftime("%H:%M:%S", time.localtime()) + " :[" + str(i) + "] dataBack(OUT) = (" + dataBack + ")\n"
            while (dataBack != "COMPLETED" and dataBack != "MOVING"):
                dataBack = self.arduino.readUntilData()
                log += "        " + time.strftime("%H:%M:%S", time.localtime()) + " :[" + str(i) + "] dataBack(IN) = (" + dataBack + ")\n"
            i += 1
            self.currentPos = vCoord
            log += "    " + time.strftime("%H:%M:%S", time.localtime()) + " : LOOP \n\n"
        log += "    " + time.strftime("%H:%M:%S", time.localtime()) + " : END \n\n"
        # For logging and debugging
        if (self.outputVals):
            print(log)
        return

    def moveTo(self, vCoord):
        log = ""
        log += "    " + time.strftime("%H:%M:%S", time.localtime()) + " : moveTo(" + str(vCoord[0]) + "," + str(vCoord[1]) +")\n"
        steps = self.pixelsToSteps(vCoord)
        log += "    " + time.strftime("%H:%M:%S", time.localtime()) + " : writeToArduino(" + str(steps[0]) + "," + str(steps[1]) + ",NO DRAW)\n"
        self.arduino.writeToArduino(str(steps[0]) + "," + str(steps[1]) + "," + str(0))
        dataBack = self.arduino.readUntilData()
        log += "    " + time.strftime("%H:%M:%S", time.localtime()) + " : dataBack(OUT) = (" + dataBack + ")\n"
        while (dataBack != "COMPLETED" and dataBack != "MOVING"):
            dataBack = self.arduino.readUntilData()
            log += "        " + time.strftime("%H:%M:%S", time.localtime()) + " : dataBack(IN) = (" + dataBack + ")\n"
        self.currentVPos = vCoord
        log += "    " + time.strftime("%H:%M:%S", time.localtime()) + " : END \n\n"
        # For logging and debugging
        if (self.outputVals):
            print(log)
        return

    def drawTestGif(self, image):
        instructions = self.getInstructions(image)
        totalInstructions = 0
        for line in instructions:
            for pixel in line:
                totalInstructions += 1
        print(totalInstructions)
        im = np.zeros([self.fullCanvas[1]*self.pixelsPerCM, self.fullCanvas[0]*self.pixelsPerCM,3],dtype=np.uint8) 
        topLeft = (int(self.offset[0]*self.pixelsPerCM), int(self.offset[1]*self.pixelsPerCM))
        topRight = (int((self.correctedCanvas[0]+self.offset[0])*self.pixelsPerCM), int(self.offset[1]*self.pixelsPerCM))
        bottomLeft = (int(self.offset[0]*self.pixelsPerCM), int((self.correctedCanvas[1]+self.offset[1])*self.pixelsPerCM))
        bottomRight = (int((self.correctedCanvas[0]+self.offset[0])*self.pixelsPerCM), int((self.correctedCanvas[1]+self.offset[1])*self.pixelsPerCM))
        im[topLeft[1]:topRight[1]+1,topLeft[0]:topRight[0]+1] = (0,0,255)
        im[topRight[1]:bottomRight[1]+1,topRight[0]:bottomRight[0]+1] = (0,0,255)
        im[bottomLeft[1]-1:bottomRight[1],bottomLeft[0]-1:bottomRight[0]] = (0,0,255)
        im[topLeft[1]:bottomLeft[1]+1,topLeft[0]:bottomLeft[0]+1] = (0,0,255)
        frames = []
        for line in instructions:
            print("===============================")
            steps = self.pixelsToSteps(line[0])
            coord = (int(steps[0]/self.motorStepsPerRev * 2*3.14*self.motorWheelRad * self.pixelsPerCM),int(steps[1]/self.motorStepsPerRev * 2*3.14*self.motorWheelRad * self.pixelsPerCM))
            # im[len(im) - coord[1]][coord[0]] = (255,0,0)
            im[coord[1]][coord[0]] = (255,0,0)
            print("I:",line[0], " S:", steps, " C:", coord)
            for pixel in range(1, len(line)):
                frames.append(Image.fromarray(im))
                steps = self.pixelsToSteps(line[pixel-1])
                coord = (int(steps[0]/self.motorStepsPerRev * 2*3.14*self.motorWheelRad * self.pixelsPerCM),int(steps[1]/self.motorStepsPerRev * 2*3.14*self.motorWheelRad * self.pixelsPerCM))
                # im[len(im) - coord[1]][coord[0]] = (255,255,255)
                im[coord[1]][coord[0]] = (255,255,255)
                steps = self.pixelsToSteps(line[pixel])
                coord = (int(steps[0]/self.motorStepsPerRev * 2*3.14*self.motorWheelRad * self.pixelsPerCM),int(steps[1]/self.motorStepsPerRev * 2*3.14*self.motorWheelRad * self.pixelsPerCM))
                # im[len(im) - coord[1]][coord[0]] = (255,0,0)
                im[coord[1]][coord[0]] = (255,0,0)
                print("I:", line[pixel-1], " S:", steps, " C:", coord)
            # im[len(im) - coord[1]][coord[0]] = (255,255,255)
            im[coord[1]][coord[0]] = (255,255,255)
            print("==============================================================")
        print("SAVING")
        frames[-1].save('TestGifFinal.png')  
        print("Frames: ", len(frames))
        # frames[0].save('TestGif.gif',
        #             save_all=True,
        #             append_images=frames[1:],
        #             duration=500,
        #             loop=0)  
        frames[0].save('TestGifFast.gif',
                    save_all=True,
                    append_images=frames[1:],
                    duration=1,
                    loop=0)  
        print("DONE")
        return 

    def drawTestGifB(self, image):
        instructions = self.getInstructions(image)
        im = np.zeros([self.fullCanvas[1]*self.pixelsPerCM, self.fullCanvas[0]*self.pixelsPerCM,3],dtype=np.uint8) 
        frames = []
        print(self.fullCanvas)
        print(im.shape)
        print(self.correctedCanvas)
        print(self.offset)
        for line in instructions:
            print(line)
            im[line[0][1]][line[0][0]] = (255,0,0)
            for pixel in range(1, len(line)):
                frames.append(Image.fromarray(im))
                im[line[pixel-1][1]][line[pixel-1][0]] = (255,255,255)
                im[line[pixel][1]][line[pixel][0]] = (255,0,0)
            im[line[-1][1]][line[-1][0]] = (255,255,255)
            print("==============================================================")
        print("SAVING")
        frames[-1].save('TestGifFinal.png')  
        print("Frames: ", len(frames))
        frames[0].save('TestGif.gif',
                    save_all=True,
                    append_images=frames[1:],
                    duration=1,
                    loop=0)  
        print("DONE")
        return 

class ImageProcessor:
    @staticmethod
    def showImage(img, name="Image"):
        if (img is not None):
            cv2.imshow(name, img)
            cv2.waitKey(0)
        else:
            print("No Image")
        return

    @staticmethod
    def loadImage(location="src/images/test.png"):
        return cv2.imread(location)

    @staticmethod
    def captureImage():
        return

    @staticmethod
    def resizeImage(image, dimSize = None, fx = 1, fy = 1, maintainAspect = True, interpolation = cv2.INTER_AREA):
        imgWidth = image.shape[1]
        imgHeight = image.shape[0]
        if (dimSize is None and fx >= 0 and fy >= 0):
            if (maintainAspect):
                 if fx < fy:
                    fx = fy
                 else:
                    fy = fx
            dim = (int(imgWidth * fx), int(imgHeight * fy))
            return cv2.resize(image, dim, interpolation=interpolation), dim
        elif (len(dimSize) == 2 and dimSize[0] > 0 and dimSize[1] > 0):
            width = dimSize[0]
            height = dimSize[1]
            dim = dimSize
            if (maintainAspect):
                imgRatio = imgWidth/imgHeight
                dimRatio = width/height
                if (imgRatio >= dimRatio):
                    dim = (int(width), int(width/imgRatio))
                else:
                    dim = (int(height*imgRatio), int(height))
            return cv2.resize(image, dim, interpolation=interpolation), dim
        else:
            return image, "Invalid Params"

    @staticmethod
    def detectEdges(image, method="canny", cannyThreshold=(100, 200), sobelKernelSize=5, blurFilter=(3, 3), blurSigmaX=0, blur = True):
        if (blurFilter[0] % 2 != 1 or blurFilter[1] % 2 != 1):
            return None, "Invalid Blur Filter Kernel"
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if (blur):
            image_blur = cv2.GaussianBlur(image_gray, blurFilter, blurSigmaX)
        else: 
            image_blur = image
        match method:
            case "canny":
                edgeImg = ImageProcessor.cannyEdge(image_blur, cannyThreshold[0], cannyThreshold[1])
            case "sobelx":
                edgeImg = ImageProcessor.sobelEdge(image_blur, 1, 0, sobelKernelSize)
            case "sobely":
                edgeImg = ImageProcessor.sobelEdge(image_blur, 0, 1, sobelKernelSize)
            case "sobelxy":
                edgeImg = ImageProcessor.sobelEdge(image_blur, 1, 1, sobelKernelSize)
            case _:
                return None, "Invalid Method"
        return edgeImg

    @staticmethod
    def cannyEdge(img, t1, t2):
        return cv2.Canny(image=img, threshold1=t1, threshold2=t2)

    @staticmethod
    def sobelEdge(img, x, y, k):
        return cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=x, dy=y, ksize=k)

    @staticmethod
    def generateLineSequence(edgeImage):
        Image.fromarray(edgeImage).save('EdgeImg.png')  
        #edgeImage[y][x]
        #checkedPixels[y][x]
        # Add 1 px empty boundary so ease error checking in recursion
        edgeImage = cv2.copyMakeBorder(edgeImage, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        linesArr = []
        line = []
        height = edgeImage.shape[0]
        width = edgeImage.shape[1]
        checkedPixels = np.zeros((height,width), dtype=bool)
        
        # Finds all the lines and adds them to an array
        # Ignores 1 px boundary we added
        for y in range (1,height-1):
            for x in range (1,width-1):
                currentPixel = (x,y)
                if (checkedPixels[y][x] == False and edgeImage[y][x] > 0):
                    line = ImageProcessor.traceLine(currentPixel, edgeImage, checkedPixels)
                    linesArr.append(line.copy())
                    line.clear()
        # Sort array longest lines to shortest lines
        sortedLinesArr = sorted(linesArr, key=len, reverse = True)

        la = "["
        for l in linesArr:
            la += str(len(l)) + ", "
        la += "]"

        sla = "["
        for l in sortedLinesArr:
            sla += str(len(l)) + ", "
        sla += "]"

        # print(la)
        # print(sla)

        return sortedLinesArr

    @staticmethod
    def traceLineR(currentPixel, image, checkedPixels, i):
        currx = currentPixel[0]
        curry = currentPixel[1]
        checkedPixels[curry][currx] = True
        line = [currentPixel]
        for x in range(-1,2):
            for y in range(-1,2):
                if (checkedPixels[curry + y][currx + x] == False and image[curry + y][currx + x] > 0):
                    line += ImageProcessor.traceLineR((currx + x, curry + y), image, checkedPixels, i+1)
                elif (image[curry + y][currx + x] == 0):
                    checkedPixels[curry + y][currx + x] = True
        return line

    @staticmethod
    def traceLine(startPixel, image, checkedPixels):
        checkPattern = [(-1,-1), (0,-1), (1,-1), (1,0), (1,1), (0,1), (-1,1), (-1,0)]
        inStack = np.zeros(image.shape, dtype=bool)
        foundEndPoint = False
        currPixel = startPixel
        line = [currPixel]
        recursiveStack = [currPixel]
        while (len(recursiveStack) != 0):
            i=0
            lineExtended = False
            prevx = line[-1][0]
            prevy = line[-1][1]
            currPixel = recursiveStack.pop(0)
            currx = currPixel[0]
            curry = currPixel[1]
            inStack[curry][currx] = False
            checkedPixels[curry][currx] = True
            line.append(currPixel)
            print("\nStart: ", startPixel, "Prev: (", prevx, ",", prevy, ") Curr: (", currx, ",", curry, ")")
            for coord in checkPattern:
                coordx = coord[0]
                coordy = coord[1]
                if ((inStack[curry + coordy][currx + coordx] == True)):
                    pixel = recursiveStack.pop(i)
                    print("    Popped: ", pixel)
                    print("        Check: (", currx + coordx, ",", curry + coordy, ") inStack:", inStack[curry + coordy][currx + coordx])
                    if (abs(currx-prevx) == 1 and abs(curry-prevy) == 1):
                        line.insert(-1,pixel)
                        print("        inStack: insert")
                    elif (prevx == startPixel[0] and prevy == startPixel[1]):
                        line.insert(0,pixel)
                        startPixel = line[0]
                        print("        inStack: insert front")
                    else:
                        line.append(pixel)
                        print("        inStack: append")
                    print("       ", line[-3], line[-2], line[-1])
                    inStack[curry + coordy][currx + coordx] = False
                    lineExtended = True
                if (checkedPixels[curry + coordy][currx + coordx] == False and image[curry + coordy][currx + coordx] > 0):
                    print("        Check: (", currx + coordx, ",", curry + coordy, ") inStack:", inStack[curry + coordy][currx + coordx])
                    print("        addedToStack")
                    recursiveStack.insert(i, (currx + coordx, curry + coordy))
                    inStack[curry + coordy][currx + coordx] = True
                    i+=1
                    lineExtended = True
                checkedPixels[curry + coordy][currx + coordx] = True

            # No new pixel found in this direction, last pixel is an end point
            # Reverse our current line so the end is the start, and new pixels get added on
            if (lineExtended == False and foundEndPoint == False):
                circular = False
                for coord in checkPattern:
                    coordx = coord[0]
                    coordy = coord[1]
                    # Circular line, no need to reverse
                    if (startPixel[0] == currx + coordx and startPixel[1] == curry + coordy):
                        circular = True
                if (not circular):
                    print("REVERSE")
                    foundEndPoint = True
                    line.reverse()
            print("Start: ", startPixel, "Prev: (", prevx, ",", prevy, ") Curr: (", currx, ",", curry, ")\n")
        line.pop(0)
        return line

    @staticmethod
    def getDrawInstructions(location = None, image = None, resizeDims=None):
        if (location is not None and image is None):
            # get picture, take it with a cam?
            image = ImageProcessor.loadImage(location=location)

        if (resizeDims is not None):
            image, newDims = ImageProcessor.resizeImage(image, dimSize = resizeDims)
        else:
            newDims = (image.shape[1],image.shape[0])

        # detect edges
        edgeImg = ImageProcessor.detectEdges(image)
        
        # parse and make lines
        instructions = ImageProcessor.generateLineSequence(edgeImg)
        
        return instructions, newDims

class SimDrawer:
    def __init__(self, width, height, startPos = (0,0), createCopies = True):
        self.position = (0,0)
        self.__canvas = (width, height)
        self.__canvasPixels = []
        self.__copiesEnabled = True if createCopies else False
        self.simSetup()
        self.setStartParams(startPos = startPos, topMotorRot=45, centerMotorRot=45, bottomMotorRot=45)
        return

    def simSetup(self, boardWidth = 20, boardSideThickness = 20, centerGap = 3, centerAxisThickness = 3, motorBaseRadius = 25, motorBaseLength = 25):
        w = self.__canvas[0]
        h = self.__canvas[1]
        xAxis = vector(1,0,0)
        yAxis = vector(0,1,0)
        zAxis = vector(0,0,1)
        # Canvas
        vCanvas = canvas(title="Simulation", width = w+100, height = h+100, ambient=vector(1,1,1))


        # Whiteboard
        center = box(color = color.black, length = w, height = h, width = boardWidth-10, pos=vector(0,0,-0.5*10))
        sideLeft = box(color = color.white, length = boardSideThickness, height = h, width = boardWidth, pos=vector(-0.5*(w+boardSideThickness),0,0))
        sideRight = box(color = color.white, length = boardSideThickness, height = h, width = boardWidth, pos=vector(0.5*(w+boardSideThickness),0,0))
        sideTop = box(color = color.white, length = w+2*boardSideThickness, height = boardSideThickness, width = boardWidth, pos=vector(0,0.5*(h+boardSideThickness),0))
        sideBottom = box(color = color.white, length = w+2*boardSideThickness, height = boardSideThickness, width = boardWidth, pos=vector(0,-0.5*(h+boardSideThickness),0))

        self.__whiteboard = compound([center, sideLeft, sideRight, sideTop, sideBottom])


        #Draw Axis
        axisWidth = boardWidth/10
        offset = centerGap+centerAxisThickness
        a = 3*centerAxisThickness+centerGap
        b = centerAxisThickness
        c = axisWidth
        centerTop = box(color = color.white, length = a, height = b, width = c, pos=vector(0,offset,0))
        centerBottom = box(color = color.white, length = a, height = b, width = c, pos=vector(0,-offset,0))
        centerLeft = box(color = color.white, length = b, height = a, width = c, pos=vector(-offset,0,0))
        centerRight = box(color = color.white, length = b, height = a, width = c, pos=vector(offset,0,0))
        axisCenter = box(color = color.red, length = centerGap, height = centerGap, width = c)
        
        cx = axisCenter.pos.x
        cy = axisCenter.pos.y
        vertLen = h/2-cy-1.5*centerGap-centerAxisThickness
        vertOffset = 1.5*centerGap+centerAxisThickness+0.5*vertLen
        horLen = w/2+cx-1.5*centerGap-centerAxisThickness
        horOffset = 1.5*centerGap+centerAxisThickness+0.5*horLen


        self.__axisCenterBox = compound([axisCenter, centerTop, centerBottom, centerLeft, centerRight])
        self.__axisCenter = box(color = color.white, length = 1, height = 1, width = 1, pos=vector(0,0,0))
        self.__axisTop = box(color = color.white, length = centerGap, height = vertLen, width = axisWidth, pos=vector(cx,vertOffset,0))
        self.__axisBottom = box(color = color.white, length = centerGap, height = vertLen, width = axisWidth, pos=vector(cx,-vertOffset,0))
        self.__axisLeft = box(color = color.white, length = horLen, height = centerAxisThickness, width = axisWidth, pos=vector(-horOffset,cy,0))
        self.__axisRight = box(color = color.white, length = horLen, height = centerAxisThickness, width = axisWidth, pos=vector(horOffset,cy,0))


        #Motors
        motorBase = cylinder(length = motorBaseLength, radius = motorBaseRadius, pos=vector(0,-0.5*motorBaseRadius,-motorBaseLength/5), color = vector(0.2,0.2,0.2), axis=-zAxis)
        motorShaftBase = cylinder(length = motorBaseLength/5, radius = motorBaseRadius/5, pos = vector(0, 0, 0), color = vector(0.4,0.4,0.4), axis=-zAxis)
        motorBaseBox = box(length = 1.25*motorBaseRadius, height = 1.25*motorBaseRadius/2, width = 0.95*motorBaseLength, pos = vector(0,-1.3*motorBaseRadius,-0.7*motorBaseLength), color = vector(0,0,0.4))
        motorWire1 = cylinder(length = motorBaseRadius/3, radius = motorBaseRadius/10, pos = vector(-2.5*motorBaseRadius/6, -1.5*motorBaseRadius, -0.7*motorBaseLength), color = color.purple, axis=-yAxis)
        motorWire2 = cylinder(length = motorBaseRadius/3, radius = motorBaseRadius/10, pos = vector(-1.25*motorBaseRadius/6, -1.5*motorBaseRadius, -0.7*motorBaseLength), color = vector(255/255, 119/255, 255/255), axis=-yAxis)
        motorWire3 = cylinder(length = motorBaseRadius/3, radius = motorBaseRadius/10, pos = vector(0, -1.5*motorBaseRadius, -0.7*motorBaseLength), color = color.red, axis=-yAxis)
        motorWire4 = cylinder(length = motorBaseRadius/3, radius = motorBaseRadius/10, pos = vector(1.25*motorBaseRadius/6, -1.5*motorBaseRadius, -0.7*motorBaseLength), color = color.orange, axis=-yAxis)
        motorWire5 = cylinder(length = motorBaseRadius/3, radius = motorBaseRadius/10, pos = vector(2.5*motorBaseRadius/6, -1.5*motorBaseRadius, -0.7*motorBaseLength), color = color.yellow, axis=-yAxis)
        motorShaft = box(length = motorBaseRadius/8, height = motorBaseRadius/5, width = motorBaseLength/2, pos = vector(0,0,0.5*motorBaseLength/2), color = vector(0.5,0.5,0))
        motorShaftIndicator = box(length = motorBaseRadius/9, height = motorBaseRadius/10, width = motorBaseLength/6, pos = vector(0,motorBaseRadius/10,0.5*motorBaseLength/6), color = vector(0,0,0.4))

        self.__centerMotorShaft = compound([motorShaft, motorShaftIndicator], origin=vector(0,0,0))
        self.__centerMotor = compound([motorBase, motorBaseBox, motorShaftBase, motorWire1, motorWire2, motorWire3, motorWire4, motorWire5], origin=vector(0,0,0))

        self.__topMotor = self.__centerMotor.clone(pos = vector(0,0,0))
        self.__topMotorShaft = self.__centerMotorShaft.clone(pos=vector(0,0,0))

        self.__bottomMotor = self.__centerMotor.clone(pos = vector(0,0,0))
        self.__bottomMotorShaft = self.__centerMotorShaft.clone(pos=vector(0,0,0))

        if self.__copiesEnabled:
            self.__centerMotorCopy = self.__centerMotor.clone(pos = vector(0,0,0))
            self.__centerMotorShaftCopy = self.__centerMotorShaft.clone(pos=vector(0,0,0))
            self.__topMotorCopy = self.__centerMotor.clone(pos = vector(0,0,0))
            self.__topMotorShaftCopy = self.__centerMotorShaft.clone(pos=vector(0,0,0))
            self.__bottomMotorCopy = self.__centerMotor.clone(pos = vector(0,0,0))
            self.__bottomMotorShaftCopy = self.__centerMotorShaft.clone(pos=vector(0,0,0))

        self.__centerMotor.rotate(radians(90), axis=yAxis)
        self.__centerMotor.rotate(radians(90), axis=xAxis)
        self.__centerMotor.pos = vector(-self.__centerMotorShaft.width + centerLeft.pos.x, 0, self.__centerMotor.height)
        self.__centerMotorShaft.rotate(radians(90), axis=yAxis)
        self.__centerMotorShaft.rotate(radians(90), axis=xAxis)
        self.__centerMotorShaft.pos = vector(self.__centerMotor.pos.x,self.__centerMotor.pos.y,self.__centerMotor.pos.z)


        self.__topMotor.rotate(radians(90), axis=xAxis)
        self.__topMotor.pos = vector(0,h/2+self.__topMotorShaft.width,self.__topMotor.height)
        self.__topMotorShaft.rotate(radians(90), axis=xAxis)
        self.__topMotorShaft.pos = vector(self.__topMotor.pos.x,self.__topMotor.pos.y,self.__topMotor.pos.z)

        self.__bottomMotor.rotate(radians(90), axis=xAxis)
        self.__bottomMotor.rotate(radians(180), axis=zAxis)
        self.__bottomMotor.pos = vector(0,-h/2-self.__bottomMotorShaft.width,self.__bottomMotor.height)
        self.__bottomMotorShaft.rotate(radians(180), axis=yAxis)
        self.__bottomMotorShaft.rotate(radians(90), axis=xAxis)
        self.__bottomMotorShaft.pos = vector(self.__bottomMotor.pos.x,self.__bottomMotor.pos.y,self.__bottomMotor.pos.z)

        if self.__copiesEnabled:
            self.__centerMotorCopy.rotate(radians(90), axis=zAxis)
            self.__centerMotorCopy.pos = vector(-2*self.__centerMotorCopy.height + centerLeft.pos.x, 0, self.__centerMotor.height)
            self.__centerMotorShaftCopy.rotate(radians(90), axis=zAxis)
            self.__centerMotorShaftCopy.pos = vector(self.__centerMotorCopy.pos.x,self.__centerMotorCopy.pos.y,self.__centerMotorCopy.pos.z)
            self.__topMotorCopy.pos = vector(0,h/2+2*self.__topMotorCopy.height,self.__topMotor.height)
            self.__topMotorShaftCopy.pos = vector(self.__topMotorCopy.pos.x,self.__topMotorCopy.pos.y,self.__topMotorCopy.pos.z)
            self.__bottomMotorCopy.rotate(radians(180), axis=zAxis)
            self.__bottomMotorShaftCopy.rotate(radians(180), axis=zAxis)
            self.__bottomMotorCopy.pos = vector(0,-h/2-2*self.__bottomMotorCopy.height,self.__bottomMotor.height)
            self.__bottomMotorShaftCopy.pos = vector(self.__bottomMotorCopy.pos.x,self.__bottomMotorCopy.pos.y,self.__bottomMotorCopy.pos.z)
        return 

    def setStartParams(self, startPos = (0,0), topMotorRot = 0, centerMotorRot = 0, bottomMotorRot = 0):
        cx = self.__axisCenter.pos.x
        cy = self.__axisCenter.pos.y
        w = self.__canvas[0]
        h = self.__canvas[1]
        coords = (startPos[0]-w/2, -(startPos[1]-h/2))
        dx = coords[0] - cx
        dy = coords[1] - cy
        
        # self.moveHor(dx)
        # self.moveVert(dy)
        self.__axisCenter.pos = vector(coords[0], coords[1], self.__axisCenter.pos.z)
        self.__axisCenterBox.pos = vector(coords[0], coords[1], self.__axisCenterBox.pos.z)
        self.__axisLeft.length = self.__axisLeft.length + dx
        self.__axisLeft.pos = vector(self.__axisLeft.pos.x + dx/2, self.__axisLeft.pos.y + dy, self.__axisLeft.pos.z)
        self.__axisRight.length = self.__axisRight.length - dx
        self.__axisRight.pos = vector(self.__axisRight.pos.x + dx/2, self.__axisRight.pos.y + dy, self.__axisRight.pos.z)
        self.__axisTop.height = self.__axisTop.height - dy
        self.__axisTop.pos = vector(self.__axisTop.pos.x + dx, self.__axisTop.pos.y + dy/2, self.__axisTop.pos.z)
        self.__axisBottom.height = self.__axisBottom.height + dy
        self.__axisBottom.pos = vector(self.__axisBottom.pos.x + dx, self.__axisBottom.pos.y + dy/2, self.__axisBottom.pos.z)

        self.__topMotor.pos = vector(dx, self.__topMotor.pos.y, self.__topMotor.pos.z)
        self.__topMotorCopy.pos = vector(dx, self.__topMotorCopy.pos.y, self.__topMotorCopy.pos.z)
        self.__topMotorShaft.pos = vector(dx, self.__topMotor.pos.y, self.__topMotor.pos.z)
        self.__topMotorShaftCopy.pos = vector(dx, self.__topMotorCopy.pos.y, self.__topMotorCopy.pos.z)
        self.__topMotorShaft.rotate(radians(topMotorRot), axis=vector(0,-1,0))
        self.__topMotorShaftCopy.rotate(radians(topMotorRot), axis=vector(0,0,1))

        self.__centerMotor.pos = vector(dx + self.__centerMotor.pos.x, dy, self.__centerMotor.pos.z)
        self.__centerMotorCopy.pos = vector(dx + self.__centerMotorCopy.pos.x, dy, self.__centerMotorCopy.pos.z)
        self.__centerMotorShaft.pos = vector(dx + self.__centerMotorShaft.pos.x, dy, self.__centerMotor.pos.z)
        self.__centerMotorShaftCopy.pos = vector(dx + self.__centerMotorShaftCopy.pos.x, dy, self.__centerMotorCopy.pos.z)
        self.__centerMotorShaft.rotate(radians(centerMotorRot), axis=vector(1,0,0))
        self.__centerMotorShaftCopy.rotate(radians(centerMotorRot), axis=vector(0,0,1))
        
        self.__bottomMotor.pos = vector(dx, self.__bottomMotor.pos.y, self.__bottomMotor.pos.z)
        self.__bottomMotorCopy.pos = vector(dx, self.__bottomMotorCopy.pos.y, self.__bottomMotorCopy.pos.z)
        self.__bottomMotorShaft.pos = vector(dx, self.__bottomMotor.pos.y, self.__bottomMotor.pos.z)
        self.__bottomMotorShaftCopy.pos = vector(dx, self.__bottomMotorCopy.pos.y, self.__bottomMotorCopy.pos.z)
        self.__bottomMotorShaft.rotate(radians(bottomMotorRot), axis=vector(0,1,0))
        self.__bottomMotorShaftCopy.rotate(radians(bottomMotorRot), axis=vector(0,0,1))

        return

    def draw(self, image):

        # parse and make lines
        instructions, newDims = ImageProcessor.getDrawInstructions(image = image, resizeDims = self.__canvas)

        offsetx = (self.__canvas[0] - newDims[0])/2
        offsety = (self.__canvas[1] - newDims[1])/2
        for line in instructions:
            self.moveTo(line[0], offset=(offsetx,offsety))
            self.drawLine(line, offset=(offsetx,offsety))
        self.moveTo((0,0))
        return

    def drawLine(self, line, offset = (0,0)):
        # print(line)
        for coord in line:
            rate(60)
            newCoord = ((coord[0] + offset[0]) - self.__canvas[0]/2 , -((coord[1] + offset[1]) - self.__canvas[1]/2))
            currentCoord = (self.__axisCenter.pos.x,self.__axisCenter.pos.y)
            self.moveHor(newCoord[0] - currentCoord[0])
            self.moveVert(newCoord[1] - currentCoord[1])
            centerPos = self.__axisCenter.pos
            self.__canvasPixels.append(box(length=1, width=1, height=1,pos=vector(centerPos.x, centerPos.y, centerPos.z)))
        return


    def moveTo(self, coords, offset = (0,0)):
        cx = self.__axisCenter.pos.x
        cy = self.__axisCenter.pos.y
        w = self.__canvas[0]
        h = self.__canvas[1]
        coords = ((coords[0] + offset[0])-w/2, -((coords[1] + offset[1])-h/2))
        if coords[0] == cx and coords[1] == cy:
            return

        deltaX = abs(coords[0]-cx)
        deltaY = abs(coords[1]-cy)
        anglePerStep = self.Motor.speed*360/self.Motor.stepsPerRev
        stepsPerPixel = self.Motor.stepsPerPixel/self.Motor.speed
        topMotorSteps = stepsPerPixel * deltaX
        centerMotorSteps = stepsPerPixel * deltaY

        if deltaX < deltaY:
            slope = abs((1.0*coords[0]-cx)/(1.0*coords[1]-cy))
            xDir = slope/stepsPerPixel if coords[0] > cx else -slope/stepsPerPixel
            yDir = 1/stepsPerPixel if coords[1] > cy else -1/stepsPerPixel
        else:
            slope = abs((1.0*coords[1]-cy)/(1.0*coords[0]-cx))
            xDir = 1/stepsPerPixel if coords[0] > cx else -1/stepsPerPixel
            yDir = slope/stepsPerPixel if coords[1] > cy else -slope/stepsPerPixel
        
        stepX = 0
        stepY = 0
        while abs(stepX - topMotorSteps) > 0.05 or abs(stepY-centerMotorSteps) > 0.05:
            rate(60)
            if (stepX > topMotorSteps-1):
                xDir = coords[0] - self.__axisCenter.pos.x
            if (stepY > centerMotorSteps-1):
                yDir = coords[1] - self.__axisCenter.pos.y
            if (stepX < topMotorSteps):
                self.moveHor(xDir)
            if (stepY < centerMotorSteps):
                self.moveVert(yDir)
            stepX += abs(stepsPerPixel*xDir)
            stepY += abs(stepsPerPixel*yDir)
        return

    def moveVert(self, val):
        anglePerStep = self.Motor.speed*360/self.Motor.stepsPerRev
        self.__centerMotor.pos = vector(self.__centerMotor.pos.x, self.__centerMotor.pos.y + val, self.__centerMotor.pos.z)
        self.__centerMotorShaft.pos = vector(self.__centerMotorShaft.pos.x, self.__centerMotorShaft.pos.y + val, self.__centerMotorShaft.pos.z) 
        self.__centerMotorShaft.rotate(radians(val*anglePerStep), axis=vector(-1,0,0)) 
        self.__centerMotorCopy.pos = vector(self.__centerMotorCopy.pos.x, self.__centerMotorCopy.pos.y + val, self.__centerMotorCopy.pos.z)
        self.__centerMotorShaftCopy.pos = vector(self.__centerMotorShaftCopy.pos.x , self.__centerMotorShaftCopy.pos.y + val, self.__centerMotorShaftCopy.pos.z) 
        self.__centerMotorShaftCopy.rotate(radians(val*anglePerStep), axis=vector(0,0,-1))

        self.__axisCenter.pos = vector(self.__axisCenter.pos.x, self.__axisCenter.pos.y + val, self.__axisCenter.pos.z)
        self.__axisCenterBox.pos = vector(self.__axisCenterBox.pos.x, self.__axisCenterBox.pos.y + val, self.__axisCenterBox.pos.z)
        self.__axisLeft.pos = vector(self.__axisLeft.pos.x, self.__axisLeft.pos.y + val, self.__axisLeft.pos.z)
        self.__axisRight.pos = vector(self.__axisRight.pos.x, self.__axisRight.pos.y + val, self.__axisRight.pos.z)
        self.__axisTop.height = self.__axisTop.height - val
        self.__axisTop.pos = vector(self.__axisTop.pos.x, self.__axisTop.pos.y + val/2, self.__axisTop.pos.z)
        self.__axisBottom.height = self.__axisBottom.height + val
        self.__axisBottom.pos = vector(self.__axisBottom.pos.x, self.__axisBottom.pos.y + val/2, self.__axisBottom.pos.z)
        return

    def moveHor(self, val):
        anglePerStep = self.Motor.speed*360/self.Motor.stepsPerRev
        self.__topMotor.pos = vector(self.__topMotor.pos.x + val, self.__topMotor.pos.y, self.__topMotor.pos.z)
        self.__topMotorShaft.pos = vector(self.__topMotorShaft.pos.x + val, self.__topMotorShaft.pos.y, self.__topMotorShaft.pos.z) 
        self.__topMotorShaft.rotate(radians(val*anglePerStep), axis=vector(0,1,0)) 
        self.__topMotorCopy.pos = vector(self.__topMotorCopy.pos.x + val, self.__topMotorCopy.pos.y, self.__topMotorCopy.pos.z)
        self.__topMotorShaftCopy.pos = vector(self.__topMotorShaftCopy.pos.x + val, self.__topMotorShaftCopy.pos.y, self.__topMotorShaftCopy.pos.z) 
        self.__topMotorShaftCopy.rotate(radians(val*anglePerStep), axis=vector(0,0,-1)) 

        self.__bottomMotor.pos = vector(self.__bottomMotor.pos.x + val, self.__bottomMotor.pos.y, self.__bottomMotor.pos.z)
        self.__bottomMotorShaft.pos = vector(self.__bottomMotorShaft.pos.x + val, self.__bottomMotorShaft.pos.y, self.__bottomMotorShaft.pos.z) 
        self.__bottomMotorShaft.rotate(radians(-val*anglePerStep), axis=vector(0,-1,0)) 
        self.__bottomMotorCopy.pos = vector(self.__bottomMotorCopy.pos.x + val, self.__bottomMotorCopy.pos.y, self.__bottomMotorCopy.pos.z)
        self.__bottomMotorShaftCopy.pos = vector(self.__bottomMotorShaftCopy.pos.x + val, self.__bottomMotorShaftCopy.pos.y, self.__bottomMotorShaftCopy.pos.z) 
        self.__bottomMotorShaftCopy.rotate(radians(-val*anglePerStep), axis=vector(0,0,-1))
        

        self.__centerMotor.pos = vector(self.__centerMotor.pos.x + val, self.__centerMotor.pos.y, self.__centerMotor.pos.z)
        self.__centerMotorShaft.pos = vector(self.__centerMotorShaft.pos.x + val, self.__centerMotorShaft.pos.y, self.__centerMotorShaft.pos.z) 
        self.__centerMotorCopy.pos = vector(self.__centerMotorCopy.pos.x + val, self.__centerMotorCopy.pos.y, self.__centerMotorCopy.pos.z)
        self.__centerMotorShaftCopy.pos = vector(self.__centerMotorShaftCopy.pos.x + val, self.__centerMotorShaftCopy.pos.y, self.__centerMotorShaftCopy.pos.z) 

        
        self.__axisCenter.pos = vector(self.__axisCenter.pos.x + val, self.__axisCenter.pos.y, self.__axisCenter.pos.z)
        self.__axisCenterBox.pos = vector(self.__axisCenterBox.pos.x + val, self.__axisCenterBox.pos.y, self.__axisCenterBox.pos.z)
        self.__axisLeft.length = self.__axisLeft.length + val
        self.__axisLeft.pos = vector(self.__axisLeft.pos.x + val/2, self.__axisLeft.pos.y, self.__axisLeft.pos.z)
        self.__axisRight.length = self.__axisRight.length - val
        self.__axisRight.pos = vector(self.__axisRight.pos.x + val/2, self.__axisRight.pos.y, self.__axisRight.pos.z)
        self.__axisTop.pos = vector(self.__axisTop.pos.x + val, self.__axisTop.pos.y, self.__axisTop.pos.z)
        self.__axisBottom.pos = vector(self.__axisBottom.pos.x + val, self.__axisBottom.pos.y, self.__axisBottom.pos.z)
        return


    def showCopies(self):
        self.__topMotorCopy.visible = True
        self.__topMotorShaftCopy.visible = True
        self.__centerMotorCopy.visible = True
        self.__centerMotorShaftCopy.visible = True
        self.__bottomMotorCopy.visible = True
        self.__bottomMotorShaftCopy.visible = True
        return

    def hideCopies(self):
        self.__topMotorCopy.visible = False
        self.__topMotorShaftCopy.visible = False
        self.__centerMotorCopy.visible = False
        self.__centerMotorShaftCopy.visible = False
        self.__bottomMotorCopy.visible = False
        self.__bottomMotorShaftCopy.visible = False
        return

    class Motor:
        speed = 60
        stepsPerRev = 2048
        stepsPerPixel = 8

def drawLinesImageSim(sortedLinesArray, imgDimensions, createGif=False):
    imgarr = []
    c = 0
    newMonoImg = np.zeros((imgDimensions[0], imgDimensions[1]), np.uint8)
    brushSize = (3,3)
    brushx = (math.ceil(brushSize[0]/2) - brushSize[0], brushSize[0] - math.ceil(brushSize[0]/2) + 1)
    brushy = (math.ceil(brushSize[1]/2) - brushSize[1], brushSize[1] - math.ceil(brushSize[1]/2) + 1)
    for line in sortedLinesArray:
        for pixel in line:
            x = pixel[0]
            y = pixel[1]
            by1 = y-brushy[0] if y-brushy[0] >= 0 else 0
            by2 = y+brushy[1] if y+brushy[1] < imgDimensions[0] else imgDimensions[0]-1
            bx1 = x-brushx[0] if x-brushx[0] >= 0 else 0
            bx2 = x+brushx[1] if x+brushx[1] < imgDimensions[1] else imgDimensions[1]-1
            newMonoImg[by1:by2+1,bx1:bx2+1] = 255
            c+=1
            if (createGif and c%100 == 0):
                imgarr.append(newMonoImg.copy())
    if (createGif):
        print("creating gif")
        imgarr = [Image.fromarray(img) for img in imgarr]
        imgarr[0].save("array2DS.gif", save_all=True, append_images=imgarr[1:], duration=1, loop=0)
        print("saved")
    return newMonoImg

def mainSim(imgLocation):
    sim = SimDrawer(1270, 720, startPos=(0,0))
    print("generating inctructions")
    
    # get picture
    image = ImageProcessor.loadImage(location=imgLocation)
    print(ImageProcessor.getDrawInstructions(image=image))
    print(image.shape)
    sim.draw(image)

    while True:
        pass
    return

def main(imgLocation):
    canvasInCM = (53, 34)
    canvasInCM = (12, 8)
    drawableCanvasInCM = (13, 8)
    motorStepsPerRev = 4096
    motorWheelRadius = 1.5
    pixelsPerCM = 20
    arduino = Arduino("com3", 9600)
    # arduino = None
    drawer = Drawer(arduino, drawableCanvasInCM, motorStepsPerRev, motorWheelRadius, pixelsPerCM)
    # drawer = Drawer(arduino, drawableCanvasInCM, motorStepsPerRev, motorWheelRadius)

    img = ImageProcessor.loadImage(location=imgLocation)
    # drawer.drawTestGif(img)
    # drawer.testDrawLine()
    drawer.drawImg(img)
    return

def mainTesting():
    canvasInCM = (53, 34)
    canvasInCM = (15, 8)
    drawableCanvasInCM = (13, 8)
    motorStepsPerRev = 4096
    motorWheelRadius = 1.5
    arduino = Arduino("com3", 9600)
    drawer = Drawer(arduino, drawableCanvasInCM, motorStepsPerRev, motorWheelRadius)

    runTestSequence(drawer, drawableCanvasInCM)
    return

def runTestSequence(drawer, vCanvas):
    drawer.recalcParams(vCanvas)
    drawer.moveTo((vCanvas[0],0))
    drawer.moveTo((vCanvas[0],vCanvas[1]))
    drawer.moveTo((0,vCanvas[1]))
    drawer.moveTo((vCanvas[0]//2,vCanvas[1]//2))
    drawer.moveTo((vCanvas[0],vCanvas[1]))

    drawer.moveTo((1,1))
    drawer.moveTo((vCanvas[0]-1,1))
    drawer.moveTo((vCanvas[0]-1,vCanvas[1]-1))
    drawer.moveTo((1,vCanvas[1]-1))
    drawer.moveTo((vCanvas[0]//2,vCanvas[1]//2))
    drawer.moveTo((vCanvas[0]-1,vCanvas[1]-1))
    drawer.moveTo((0,0))


if __name__ == '__main__':
    imgLocation = "./src/images/test3.png"
    simMode = True
    simMode = False
    testMode = True
    testMode = False
    if (simMode):
        mainSim(imgLocation)
    elif (testMode):
        mainTesting()
    else:
        main(imgLocation)