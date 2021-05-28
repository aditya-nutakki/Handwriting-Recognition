from __future__ import division
from __future__ import print_function

import sys
import argparse
import os
import cv2
import numpy
import math
from imutils.object_detection import non_max_suppression
import editdistance
from DataLoader import DataLoader, Batch
from Model import Model, DecoderType
from SamplePreprocessor import preprocess


class FilePaths:
    "filenames and paths to data"
    fnCharList = '../model/charList.txt'
    fnAccuracy = '../model/accuracy.txt'
    fnTrain = '../data/'
    # fnInfer = '../data/test/1.jpeg'  #ORIGINAL ie HOW IT WAS BEFORE USING IT
    fnInfer = '../data/output'
    fnCorpus = '../data/corpus.txt'


#IMPORTANT THINGS HAPPEN HERE --> TILL "train" function


# SCALES UP/DOWN ANY IMAGE TO (320x320)px
def scale(img):
    return cv2.resize(img, (320, 320))


# SCALES ONLY THE ROI 2x
def roi_scale(img):
    try:
        return cv2.resize(img, (128,32))
    except:
        return img

# USED FOR THE VISUALIZING PURPOSES ONLY ie GIVES YOU AN IDEA WHERE THE CO-ORDINATES ARE LOCATED AND TO DRAW POINTS AT THEM

def show_boxes(img, boxes):
    for startX, startY, endX, endY in boxes:
        cv2.circle(img, (startX, startY), radius=2, color=(0, 255, 255), thickness=-1)  # shows top left co-ordinate
        # cv2.circle(img, (endX, endY), radius=2, color=(0, 0, 0), thickness=-1) #shows bottom left co-ordinate
        cv2.putText(img, text=f"{startX},{startY}", org=(startX, startY), color=(0, 0, 0), fontScale=0.4, thickness=1,
                    fontFace=cv2.FONT_HERSHEY_COMPLEX)

        print(f"({startX},{startY}) = {mse(startX, startY)}")
        print()
        cv2.imshow("Blank Image", img)
        cv2.waitKey(0)


# USED TO CHECK IF A NUMBER IS BETWEEN TWO DIFFERENT LIMITS
def between(x, n_range):
    if (x > n_range[0]) and (x < n_range[1]):
        return True

    else:
        return False


############################## THE MSE FUCNTION GIVES YOU THE VALUE SO AS TO ARRANGE IT IN READING ORDER #################

def mse(x, y):
    # Q1
    try:

        if x < 160 and y < 160:
            return math.sqrt(math.log(x, 15) + math.log(y, 2))

        # Q2
        elif x > 160 and y < 160:
            return math.sqrt(math.log(x, 13) + math.log(y, 2))

        # Q3 is split into another 4 parts

        elif x < 40 and between(y, (160, 240)):
            return math.sqrt(math.log(x, 1.7) + math.log(y, 2))

        elif x > 40 and between(y, (160, 240)):
            return math.sqrt(math.log(x, 1.45) + math.log(y, 2))

        elif x < 40 and between(y, (240, 320)):
            return math.sqrt(math.log(x, 1.1) + math.log(y, 1.35))

        elif x > 40 and between(y, (240, 320)):
            return math.sqrt(math.log(x, 1.25) + math.log(y, 2.55))

        # Q4
        else:
            return math.sqrt(math.sqrt(x) * 2 + math.log(y, 1.2))

    except:
        pass

############################################### SORT'S CO-ORDINATES FROM LEFT -> RIGHT AND TOP -> BOTTOM

def sort_boxes(box):

    box = sorted(box, key=lambda x: (x[1], x[0], x[2], x[3]))

    # MAKING A DICTIONARY SO AS TO FIND WHAT THE CO-ORDINATES GIVE WHEN BEING SENT INTO A CUSTOM MATH FUNCTION (mse)

    new_boxes = {}

    for i in range(len(box)):
        new_boxes[f"{mse(box[i][0], box[i][1])}"] = box[i]

    # print(new_boxes) #DICTIONARY
    new_boxes = sorted(new_boxes.items(), key=lambda x: x[0])

    ordered_boxes = []
    # print(type(new_boxes[0][1]))  # --> gives you the list of co-ordinates in required order
    # print(new_boxes[1][0]) #--> gives you the value after passing the x,y into the mse function

    for i in range(len(new_boxes)):
        ordered_boxes.append(new_boxes[i][1])

    # show_boxes(img, ordered_boxes)
    return ordered_boxes




def train(model, loader):
    "train NN"
    epoch = 0  # number of training epochs since start
    bestCharErrorRate = float('inf')  # best valdiation character error rate
    noImprovementSince = 0  # number of epochs no improvement of character error rate occured
    earlyStopping = 5  # stop training after this number of epochs without improvement
    while True:
        epoch += 1
        print('Epoch:', epoch)

        # train
        print('Train NN')
        loader.trainSet()
        while loader.hasNext():
            iterInfo = loader.getIteratorInfo()
            batch = loader.getNext()
            loss = model.trainBatch(batch)
            print('Batch:', iterInfo[0], '/', iterInfo[1], 'Loss:', loss)

        # validate
        charErrorRate = validate(model, loader)

        # if best validation accuracy so far, save model parameters
        if charErrorRate < bestCharErrorRate:
            print('Character error rate improved, save model')
            bestCharErrorRate = charErrorRate
            noImprovementSince = 0
            model.save()
            open(FilePaths.fnAccuracy, 'w').write(
                'Validation character error rate of saved model: %f%%' % (charErrorRate * 100.0))
        else:
            print('Character error rate not improved')
            noImprovementSince += 1

        # stop training if no more improvement in the last x epochs
        if noImprovementSince >= earlyStopping:
            print('No more improvement since %d epochs. Training stopped.' % earlyStopping)
            break


def validate(model, loader):
    "validate NN"
    print('Validate NN')
    loader.validationSet()
    numCharErr = 0
    numCharTotal = 0
    numWordOK = 0
    numWordTotal = 0
    while loader.hasNext():
        iterInfo = loader.getIteratorInfo()
        print('Batch:', iterInfo[0], '/', iterInfo[1])
        batch = loader.getNext()
        (recognized, _) = model.inferBatch(batch)

        print('Ground truth -> Recognized')
        for i in range(len(recognized)):
            numWordOK += 1 if batch.gtTexts[i] == recognized[i] else 0
            numWordTotal += 1
            dist = editdistance.eval(recognized[i], batch.gtTexts[i])
            numCharErr += dist
            numCharTotal += len(batch.gtTexts[i])
            print('[OK]' if dist == 0 else '[ERR:%d]' % dist, '"' + batch.gtTexts[i] + '"', '->',
                  '"' + recognized[i] + '"')

    # print validation result
    charErrorRate = numCharErr / numCharTotal
    wordAccuracy = numWordOK / numWordTotal
    print('Character error rate: %f%%. Word accuracy: %f%%.' % (charErrorRate * 100.0, wordAccuracy * 100.0))
    return charErrorRate


def only_numbers(x):
    count = 0
    y= []
    for i in x:
        if i.endswith(".jpeg"):
            i = i[:-5]
            y.append(i)

        else:
            continue

    return y


def str_to_int(x):

    for i in range(0, len(x)):
        x[i] = int(x[i])

    return x


def preprocess_image_dir(x):
    x = only_numbers(x)
    x = str_to_int(x)
    x.sort()

    return x

################################################# MOST IMPORTANT PART OF THE SCIRPT ########################################

def scanning(file_name , savePath):

    try:
        img = cv2.imread(f"test_images/{file_name}")
        img = scale(img)
    except:
        print("Image not found or None or wrong extension , check again")

    ############################################# EAST DETECTOR MODEL ###################################################

    net = cv2.dnn.readNet(model="frozen_east_text_detection.pb")
    blob = cv2.dnn.blobFromImage(img, scalefactor=1, size=(320, 320), mean=(123.68, 116.78, 103.94), swapRB=True,
                                 crop=False)

    outputLayers = []
    outputLayers.append("feature_fusion/Conv_7/Sigmoid")
    outputLayers.append("feature_fusion/concat_3")

    net.setInput(blob)
    output = net.forward(outputLayers)

    scores = output[0]
    geometry = output[1]

    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []


    ############################ LOOP OVER THE NUMBER OF ROWS TO FIND CO-ORDINATES OF THE LOCATED TEXT #######################

    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the geometrical
        # data used to derive potential bounding box coordinates that
        # surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability, ignore it
            if scoresData[x] < 0.15:
                continue
            # compute the offset factor as our resulting feature maps will
            # be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            # extract the rotation angle for the prediction and then
            # compute the sin and cosine
            angle = anglesData[x]
            cos = numpy.cos(angle)
            sin = numpy.sin(angle)
            # use the geometry volume to derive the width and height of
            # the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]
            # compute both the starting and ending (x, y)-coordinates for
            # the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)
            # add the bounding box coordinates and probability score to
            # our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes

    ################################################### SORTING  BOXES ######################################################

    boxes = non_max_suppression(numpy.array(rects), probs=confidences)
    boxes = sort_boxes(boxes)
    print(f"words detected  = {len(boxes)}")

    try:
        os.mkdir(f"{savePath}")

    except:
        print("Image directory already made")


    ########################### LOOPING OVER "BOXES" TO DRAW RECTANGLES AROUND DETECTED TEXT #################################
    count = 1


    for (startX, startY, endX, endY) in boxes:
        # scale the bounding box coordinates based on the respective ratios

        # GETTING THE IMAGE ALONE

        roi = img[startY:endY, startX:endX]
        roi = roi_scale(roi)

        # MARKING THE CORNER POINTS
        cv2.circle(img, (startX, startY), 2, color=(0, 255, 0), thickness=-1) #TOP RIGHT CO-ORDINATE
        # cv2.circle(img, (endX, endY), 3, color=(0, 0, 0), thickness=-1) #BOTTOM-LEFT CO-ORDINATE

        font = cv2.FONT_HERSHEY_COMPLEX

        # SAVING THE ROI IMAGE INTO THE DIRECTORY
        cv2.imwrite(f"{savePath}/{count}.jpeg", roi)
        count += 1

        # draw the bounding box on the image
        cv2.rectangle(img, (startX, startY), (endX, endY), (0, 0, 0), 1)

        cv2.putText(img, f"{startX},{startY}", (startX, startY), font, fontScale=0.42, color=(0, 0, 0), thickness=2)
        #
        # cv2.imshow("ROI", roi)
        # cv2.imshow("PRESS ANY KEY TO SEE NEXT DETECTED WORD", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    cv2.imshow("PRESS ANY KEY TO QUIT", img)
    cv2.waitKey(0)


def infer(model, fnImg):
    "recognize text in image provided by file path"

    # All images in this directory -->  ..\src\test_images

    all_images = os.listdir("test_images")
    print(all_images)


    for j in all_images:

        k = j.replace(".","") + "_text"
        #print(j)

        # try:
        #     os.mkdir(f"..\data\output\segmented_text\{k}")
        # except:
        #     pass

        segmented_dir = f"..\data\output\{k}"
        scanning(j ,segmented_dir)

        #os.chdir doesnt work for some reason , so just use this
        images = os.listdir(f"{FilePaths.fnInfer}/{k}")

        images = preprocess_image_dir(images)

        #print(images) # A LIST OF NUMBERS ie FILE NAMES which are of integer type

        # images in this case is being considered as a string - as in - its not being sorted accordingly its being sorted as 1 ,2 , 21, 22, 3, 4, etc
        # so convert them to int and sort them and then convert them back to str when passing it into fnImg

        text = ""

        fnImg = FilePaths.fnInfer

        for image in images:

            fnImg = FilePaths.fnInfer + r"/" + f"{k}/" f"{str(image)}" + ".jpeg"

            if fnImg.endswith(".jpeg"):
                #print(fnImg)  # THIS IS THE IMAGE PATH AND NOT AN IMAGE looks something like this 		../data/test/1.jpeg
                # print(type(fnImg)) # TYPE STRING
                img = preprocess(cv2.imread(fnImg, cv2.IMREAD_GRAYSCALE), Model.imgSize)
                # cv2.imshow("INSIDE INFER FUNCTION ", img)
                # cv2.waitKey(0)

                batch = Batch(None, [img])

                try:
                    (recognized, probability) = model.inferBatch(batch, True)
                    print('Recognized:', '"' + recognized[0] + '"')

                except:
                    continue

                text += recognized[0] + " "
                # print('Probability:', probability[0])

            else:
                print("something's wrong with directory , please check accordingly")


        print(text)
        # SAVING THE FILE AS A .TXT
        f = open(f"{FilePaths.fnInfer}/{k}/recongnized_text.txt", "w")
        f.write(text)
        f.close()
        print()



def main():
    "main function"
    # optional command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', help='train the NN', action='store_true')
    parser.add_argument('--validate', help='validate the NN', action='store_true')
    parser.add_argument('--beamsearch', help='use beam search instead of best path decoding', action='store_true')
    parser.add_argument('--wordbeamsearch', help='use word beam search instead of best path decoding',
                        action='store_true')
    parser.add_argument('--dump', help='dump output of NN to CSV file(s)', action='store_true')

    args = parser.parse_args()

    decoderType = DecoderType.BestPath
    if args.beamsearch:
        decoderType = DecoderType.BeamSearch
    elif args.wordbeamsearch:
        decoderType = DecoderType.WordBeamSearch

    # train or validate on IAM dataset
    if args.train or args.validate:
        # load training data, create TF model
        loader = DataLoader(FilePaths.fnTrain, Model.batchSize, Model.imgSize, Model.maxTextLen)

        # save characters of model for inference mode
        open(FilePaths.fnCharList, 'w').write(str().join(loader.charList))

        # save words contained in dataset into file
        open(FilePaths.fnCorpus, 'w').write(str(' ').join(loader.trainWords + loader.validationWords))

        # execute training or validation
        if args.train:
            model = Model(loader.charList, decoderType)
            train(model, loader)
        elif args.validate:
            model = Model(loader.charList, decoderType, mustRestore=True)
            validate(model, loader)

    # infer text on test image

    else:
        print(open(FilePaths.fnAccuracy).read())
        model = Model(open(FilePaths.fnCharList).read(), decoderType, mustRestore=True, dump=args.dump)
        infer(model, FilePaths.fnInfer)


if __name__ == '__main__':
    main()


