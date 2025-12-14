from mlops.infer import predict

def predict_emotion(frame_bgr, model_key: str):
    return predict(frame_bgr, model_key=model_key)
