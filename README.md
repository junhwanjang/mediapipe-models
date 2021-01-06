# mediapipe-models
Google Mediapipe public TFLite models implemented using Tensorflow-keras (https://mediapipe.dev)

### Requirements
- Tensorflow - 1.13.1

### Implementation process
1. Hand Tracking
    * Related links
        - [Google AI blog post: On device real time hand tracking](https://ai.googleblog.com/2019/08/on-device-real-time-hand-tracking-with.html)
        - Github: [Mediapipe: Hand tracking](https://github.com/google/mediapipe/blob/master/mediapipe/docs/hand_tracking_mobile_gpu.md)

    * Hand Landmark 2D,3D
        - Model Architecture (OK)
        - Set Pretrained weights (OK)
        - Convert TFLite model for 4 channels input (OK)
    
    * Palm(Hand) Detection
        - Model Architecture (OK)
        - Set Pretrained weights (OK)
        - Convert TFLite model for 4 channels input (OK)

2. Face Detection
    * Related links
        - Paper: ["BlazeFace: Sub-millisecond Neural Face Detection on Mobile GPUs"](https://arxiv.org/abs/1907.05047)
        - Github: [Mediapipe: Face detection](https://github.com/google/mediapipe/blob/master/mediapipe/docs/face_detection_mobile_gpu.md)

    * Face Detection
        - Model Architecture (OK)
        - Set Pretrained weights (TODO)
        - Convert TFLite model for 4 channels input (TODO)

3. Hair Segmentation
    * Related links
        - Paper: ["Real-time Hair segmentation and recoloring on Mobile GPUs"](https://arxiv.org/abs/1907.06740)
        - Github: [Mediapipe: Hair segmentation](https://github.com/google/mediapipe/blob/master/mediapipe/docs/hair_segmentation_mobile_gpu.md)

    * Hair Segmentation
        - Model Architecture (TODO)
        - Set Pretrained weights (TODO)
        - Convert TFLite model for 4 channels input (TODO)
