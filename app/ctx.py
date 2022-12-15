from dnn.openvino.EfficientFER import EfficientFER
from dnn.openvino.FaceDetectionRetail0044 import FaceDetectionRetail0044

face_detector = FaceDetectionRetail0044('models/face-detection-retail-0044.xml')

emotion_recognizer = EfficientFER(
    'models/EfficientFER5_96x96.xml',
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
