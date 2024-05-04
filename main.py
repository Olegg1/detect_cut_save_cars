from roboflow import Roboflow

rf = Roboflow(api_key="4RPgmU9QIwvUoV7dx4lp")
project = rf.workspace().project("car-qe6jr")
model = project.version("1").model

job_id, signed_url, expire_time = model.predict_video(
    "C:\\Users\\User\\Downloads\\DeepSort\\BMW vs MERCEDES #shorts.mp4",
    fps=5,
    prediction_type="batch-video"
)

results = model.poll_until_video_results(job_id)

if results['predictions']:
    xmin, ymin, xmax, ymax = int(results['predictions'][0]), int(results['predictions'][1]), int(results['predictions'][2]), int(results['predictions'][3])
