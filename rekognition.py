#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

# start your camera using photobooth for a preview and to warm up the camera before running this script

import json
import os
from sys import argv

import boto3
import cv2
import inflect
import numpy as np
from botocore.exceptions import BotoCoreError, ClientError
from playsound import playsound


class bcolors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'


region = 'eu-west-1'  # change this to switch to another AWS region
colors = [['lime', 0, 255, 0], ['blue', 255, 0, 0], ['red', 0, 0, 255], ['fuchsia', 255, 0, 255], ['silver', 192, 192, 192],
          ['cyan', 0, 255, 255], ['orange', 255, 99, 71], ['white', 255,
                                                           255, 255], ['black', 0, 0, 0], ['gray', 128, 128, 128],
          ['green', 0, 128, 0], ['purple', 128, 0, 128], ['navy', 0, 0, 128]]

polly = boto3.client("polly", region_name=region)
reko = boto3.client('rekognition', region_name=region)
translate = boto3.client('translate', region_name=region)
s3resource = boto3.resource('s3', region_name=region)
dynamodb = boto3.resource('dynamodb', region_name=region)

p = inflect.engine()
bucket = 'my8uck37'
collectionId = 'myCollection'
table = dynamodb.Table('ImageCollection')

# Take a photo with USB webcam
# Set save to True if you want to save the image (in the current working directory)
# and open Preview to see the image


def take_image():
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("opencv_frame")
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        ret, frame = cam.read()
        cv2.imshow("opencv_frame.jpg", frame)
        if not ret:
            break

        k = cv2.waitKey(1)
        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k % 256 == 32:
            # SPACE pressed
            img_name = "opencv_frame.jpg"
            cv2.imwrite(img_name, frame)
            break

    cam.release()
    cv2.destroyAllWindows()
    os.system('opencv_frame.jpg')
    return img_name

# Read image from file


def read_image(filename):
    try:
        fin = open(filename, 'rb')
        encoded_image_bytes = fin.read()
        fin.close()
        return encoded_image_bytes
    except IOError as e:
        print("I/O error({0}): {1}".format(e.errno, e.strerror))
        exit(-1)

# Provide a string and an optional voice attribute and play the streamed audio response
# Defaults to the Salli voice


def speak(tag, text_string, voice):
    try:
        # Request speech synthesis
        response = polly.synthesize_speech(Text=text_string,
                                           TextType="text", OutputFormat="mp3", VoiceId=voice)
    except (BotoCoreError, ClientError) as error:
        # The service returned an error, exit gracefully
        print(error)
        exit(-1)
    # Access the audio stream from the response
    if "AudioStream" in response:
        soundfile = open(str(tag) + '.mp3', 'wb')
        soundfile.write(response['AudioStream'].read())
        soundfile.close()
        playsound(str(tag) + '.mp3')
    else:
        # The response didn't contain audio data, return False
        print("Could not stream audio")
        return(False)

# Amazon Rekognition label detection


def reko_detect_labels(image_bytes):
    print("Calling Amazon Rekognition: detect_labels")
    response = reko.detect_labels(
        Image={
            'Bytes': image_bytes
        },
        MaxLabels=10,
        MinConfidence=60
    )
    print(json.dumps(response, sort_keys=True, indent=4))
    return response

# rekognition facial detection


def reko_detect_faces(image_bytes):
    print("Calling Amazon Rekognition: detect_faces")
    response = reko.detect_faces(
        Image={
            'Bytes': image_bytes
        },
        Attributes=['ALL']
    )
    print(json.dumps(response, sort_keys=True, indent=4))
    return response

# create verbal response describing the detected lables in the response from Rekognition
# there needs to be more than one lable right now, otherwise you'll get a leading 'and'


def create_verbal_response_labels(reko_response):
    mystring = "I detected the following labels: "
    humans = False
    labels = len(reko_response['Labels'])
    if labels == 0:
        mystring = "I cannot detect anything."
    else:
        i = 0
        for mydict in reko_response['Labels']:
            i += 1
            if mydict['Name'] == 'People':
                humans = True
                continue
            print("%s\t(%.2f)" % (mydict['Name'], mydict['Confidence']))
            if i < labels:
                newstring = "%s, " % (mydict['Name'].lower())
                mystring = mystring + newstring
            else:
                newstring = "and %s. " % (mydict['Name'].lower())
                mystring = mystring + newstring
            if ('Human' in mydict.values()) or ('Person' in mydict.values()):
                humans = True
    return humans, mystring


def create_verbal_response_face(reko_response):
    mystring = ""

    persons = len(reko_response['FaceDetails'])
    print("number of persons = ", persons)

    if persons == 1:
        mystring = "I can see one face. "
    else:
        mystring = "I can see %d faces. " % (persons)
    i = 0
    for mydict in reko_response['FaceDetails']:
        # Boolean True|False values for these facial features
        age_range_low = mydict['AgeRange']['Low']
        age_range_high = mydict['AgeRange']['High']
        beard = mydict['Beard']['Value']
        eyeglasses = mydict['Eyeglasses']['Value']
        eyesopen = mydict['EyesOpen']['Value']
        mouthopen = mydict['MouthOpen']['Value']
        mustache = mydict['Mustache']['Value']
        smile = mydict['Smile']['Value']
        sunglasses = mydict['Sunglasses']['Value']
        if persons == 1:
            mystring = mystring + \
                "The person is %s. " % (mydict['Gender']['Value'].lower())
        else:
            mystring = mystring + "The %s person is %s. " % (p.number_to_words(
                p.ordinal(str([i+1]))), mydict['Gender']['Value'].lower())
        if mydict['Gender']['Value'] == 'Male':
            he_she = 'he'
        else:
            he_she = 'she'
        print("Person %d (%s):" % (i+1, colors[i][0]))
        print("\tGender: %s\t(%.2f)" %
              (mydict['Gender']['Value'], mydict['Gender']['Confidence']))
        print("\tEyeglasses: %s\t(%.2f)" %
              (eyeglasses, mydict['Eyeglasses']['Confidence']))
        print("\tSunglasses: %s\t(%.2f)" %
              (sunglasses, mydict['Sunglasses']['Confidence']))
        print("\tSmile: %s\t(%.2f)" % (smile, mydict['Smile']['Confidence']))
        if eyeglasses == True and sunglasses == True:
            mystring = mystring + \
                "%s is wearing glasses. " % (he_she.capitalize(), )
        elif eyeglasses == True and sunglasses == False:
            mystring = mystring + \
                "%s is wearing spectacles. " % (he_she.capitalize(), )
        elif eyeglasses == False and sunglasses == True:
            mystring = mystring + \
                "%s is wearing sunglasses. " % (he_she.capitalize(), )
        if smile:
            true_false = 'is'
        else:
            true_false = 'is not'
        mystring = mystring + \
            "%s %s smiling. " % (he_she.capitalize(), true_false)
        if mydict['Gender']['Value'] == 'Male':
            his_her = 'his'
        else:
            his_her = 'her'
        if mouthopen:
            true_false = 'is'
        else:
            true_false = 'is not'
        mystring = mystring + \
            "%s Mouth %s open. " % (his_her.capitalize(), true_false)
        if eyesopen:
            true_false = 'open'
        else:
            true_false = 'closed'
        mystring = mystring + \
            "%s Eyes are %s. " % (his_her.capitalize(), true_false)
        if beard:
            mystring = mystring + "He has a beard. "
        if mustache:
            mystring = mystring + "He has a mustache. "
        mystring = mystring + "%s estimated age ist between %s and %s years. " % (
            his_her.capitalize(), age_range_low, age_range_high)
        print("\tEmotions:")
        j = 0
        for emotion in mydict['Emotions']:
            if j == 0:
                mystring = mystring + \
                    "%s looks %s. " % (he_she.capitalize(),
                                       emotion['Type'].lower())
            print("\t\t%s\t(%.2f)" % (emotion['Type'], emotion['Confidence']))
            j += 1
        # Find bounding box for this face
        height = mydict['BoundingBox']['Height']
        left = mydict['BoundingBox']['Left']
        top = mydict['BoundingBox']['Top']
        width = mydict['BoundingBox']['Width']
        i += 1

        if i > 12:
            break

    return mystring


def save_image_with_bounding_boxes(encoded_image, reko_response):
    encoded_image = np.frombuffer(encoded_image, np.uint8)
    image = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)
    image_height, image_width = image.shape[:2]
    i = 0
    for mydict in reko_response['FaceDetails']:
        # Find bounding box for this face
        height = mydict['BoundingBox']['Height']
        left = mydict['BoundingBox']['Left']
        top = mydict['BoundingBox']['Top']
        width = mydict['BoundingBox']['Width']
        # draw this bounding box
        image = draw_bounding_box(
            image, image_width, image_height, width, height, top, left, colors[i], i)
        i += 1
        if i > 12:
            break
    # write the image to a file
    cv2.imwrite('face_bounding_boxes.jpg', image)
    os.system('start face_bounding_boxes.jpg')

# draw bounding box around one face


def draw_bounding_box(cv_img, cv_img_width, cv_img_height, width, height, top, left, color, img_nmbr):
    # calculate bounding box coordinates top-left - x,y, bottom-right - x,y
    width_pixels = int(width * cv_img_width)
    height_pixels = int(height * cv_img_height)
    left_pixel = int(left * cv_img_width)
    top_pixel = int(top * cv_img_height)
    cv2.rectangle(cv_img, (left_pixel, top_pixel), (left_pixel+width_pixels,
                                                    top_pixel+height_pixels), (color[1], color[2], color[3]), 2)
    img_nmbr += 1
    cv2.putText(cv_img, str(img_nmbr), (left_pixel, top_pixel), cv2.FONT_HERSHEY_SIMPLEX,
                2, (color[1], color[2], color[3]), 1)
    return cv_img

# compare faces


def compare_faces():
    sourceFile = 'image6.jpg'
    targetFile = 'image6.jpg'

    imageSource = open(sourceFile, 'rb')
    imageTarget = open(targetFile, 'rb')

    response = reko.compare_faces(SimilarityThreshold=50,
                                  SourceImage={'Bytes': imageSource.read()},
                                  TargetImage={'Bytes': imageTarget.read()})

    # print json.dumps(response, sort_keys=True, indent=4)

    if not response['FaceMatches']:
        print(bcolors.RED + 'No Match')
    else:
        for faceMatch in response['FaceMatches']:
            position = faceMatch['Face']['BoundingBox']
            confidence = str(faceMatch['Face']['Confidence'])
            print(bcolors.GREEN + 'The faces matches with ' +
                  confidence + '% confidence')

    imageSource.close()
    imageTarget.close()

# Upload Image to a S3-Bucket


def upload_image(upload_image):
    # Filename in S3-Bucket
    key = upload_image
    myBucket = s3resource.Bucket(bucket)
    myBucket.upload_file(key, key)

# index_faces: Detect faces in an image and add them to a collection.


def index_faces(index_image):
    photo = index_image

    client = boto3.client('rekognition', region_name=region)

    response = client.index_faces(CollectionId=collectionId,
                                  Image={'S3Object': {
                                      'Bucket': bucket, 'Name': photo}},
                                  ExternalImageId=photo,
                                  MaxFaces=5,
                                  QualityFilter="AUTO",
                                  DetectionAttributes=['ALL'])

    print('Results for ' + photo)
    print('Faces indexed:')
    for faceRecord in response['FaceRecords']:
        print('  Face ID: ' + faceRecord['Face']['FaceId'])
        print('  Location: {}'.format(faceRecord['Face']['BoundingBox']))

    print('Faces not indexed:')
    for unindexedFace in response['UnindexedFaces']:
        print(' Location: {}'.format(
            unindexedFace['FaceDetail']['BoundingBox']))
        print(' Reasons:')
        for reason in unindexedFace['Reasons']:
            print('   ' + reason)

    return response['FaceRecords']

# delete_faces: Delete faces from a collection.


def delete_faces(face_record):
    for faceRecord in face_record:
        faces = [faceRecord['Face']['FaceId']]

    client = boto3.client('rekognition', region_name=region)

    response = client.delete_faces(CollectionId=collectionId,
                                   FaceIds=faces)

    print(str(len(response['DeletedFaces'])) + ' faces deleted:')
    for faceId in response['DeletedFaces']:
        print(faceId)

# Search for faces in a collection that match a supplied face ID


def search_faces(face_record):
    threshold = 98
    maxFaces = 4

    for faceRecord in face_record:
        client = boto3.client('rekognition', region_name=region)

        response = client.search_faces(CollectionId=collectionId,
                                       FaceId=faceRecord['Face']['FaceId'],
                                       FaceMatchThreshold=threshold,
                                       MaxFaces=maxFaces)

        print(response)
        faceMatches = response['FaceMatches']
        print('Matching faces')
        for match in faceMatches:
            print('FaceId: ' + match['Face']['FaceId'])
            print('Similarity: ' + "{:.2f}".format(match['Similarity']) + "%")
            print('ExternalImageId: ' + match['Face']['ExternalImageId'])

    return faceMatches

# Read Name from DynamoDB by Faces


def read_name_by_faces(faceMatches):
    for match in faceMatches:
        faceid = match['Face']['FaceId']
        print(faceid)
        response = table.get_item(
            Key={
                'faceid': faceid
            }
        )
        print(response)

        item = response['Item']
        name = item['first_name']
        print(item)
        print("Hello, {}" .format(name))

# START MAIN


# if no arguments take a photo
# if one argument open the image file and decode it
# if more than on argument exit gracefully and print usage guidance
if len(argv) == 1:
    image_name = take_image()
elif len(argv) == 2:
    print("opening image in file: ", argv[1])
    image_name = argv[1]
else:
    print("Use with no arguments to take a photo with the camera, or one argument to use a saved image")
    exit(-1)

encoded_image = read_image(image_name)
#upload_image(image_name)
# Import Faces to Collection
#face_record = index_faces(image_name)
#print(face_record)

# Search detected Faces from Image by FaceId
#faceMatches = search_faces(face_record)
#print(faceMatches)

# Read Items from DynamoDB by FaceId
#read_name_by_faces(faceMatches)

# Delete Faces from Collection - House Keeping
#delete_faces(face_record)

labels = reko_detect_labels(encoded_image)
humans, labels_response_string = create_verbal_response_labels(labels)
print(labels_response_string)
speak('label', labels_response_string, 'Joanna')

if humans:
    print("Detected Human: ", humans, "\n")
    reko_response = reko_detect_faces(encoded_image)

    all_faces=reko_response['FaceDetails']
    # Initialize list object
    # Crop face from image
    for face in all_faces:
        box=face['BoundingBox']

    print(box)


    faces_response_string = create_verbal_response_face(reko_response)
    save_image_with_bounding_boxes(encoded_image, reko_response)
    print(faces_response_string)
    speak('faces_en', faces_response_string, 'Joanna')

    translated_response_string = translate.translate_text(
        Text=faces_response_string, SourceLanguageCode='en', TargetLanguageCode='de')['TranslatedText']
    print(translated_response_string)
    speak('faces_translated', translated_response_string, 'Marlene')
else:
    print("No humans detected. Skipping facial recognition")
