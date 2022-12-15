from dnn.opencv.EfficientFER import EfficientFER
from dnn.opencv.FaceDetectionRetail0044 import FaceDetectionRetail0044

face_detector = FaceDetectionRetail0044(
    'models/face-detection-retail-0044.caffemodel',
    'models/face-detection-retail-0044.prototxt',
)

emotion_recognizer = EfficientFER(
    'models/EfficientFER5_96x96.onnx',
    (96, 96),
    emotions=[
        'anger',
        # 'contempt',
        # 'disgust',
        # 'fear',
        'happiness',
        'neutral',
        'sadness',
        'surprise',
    ],
)
