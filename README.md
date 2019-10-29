# aws-rekognition
A collection of lambda functions that are invoked by Amazon S3 or Amazon API Gateway to analyze uploaded images with Amazon Rekognition and tell and translate the picture labels with Polly.

### Tech Stack
#### Required Tools
* boto3
* cv2
* numpy
* aws cli
* Amazon Rekognition
* Amazaon Polly
* Amazon Translate
* Amazon Cognito
* AWS Lambda
* Amazon API Gateway
* Amazon S3
* Amazon DynamoDB

#### Starts with arguments
if no arguments take a photo
if one argument open the image file and decode it
if more than on argument exit gracefully and print usage guidance
