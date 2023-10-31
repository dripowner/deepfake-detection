import clearml
from clearml.automation import PipelineController
from clearml import Task


# Connecting ClearML with the current pipeline,
# from here on everything is logged automatically
pipe = PipelineController(
    name="Pretrained deepfake-classification model test", project="deepfake-detection", version="0.0.1", add_pipeline_tags=False
)

pipe.add_parameter(
    name='model_weights_url',
    description='url weights file', 
    default='https://github.com/selimsef/dfdc_deepfake_challenge/releases/download/0.0.1/final_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_36'
)

pipe.add_parameter(
    name='dataset_name',
    description='name of dataset', 
    default='test-deepfake-detection-0.1'
)

pipe.add_step(
    name="load_model_weights",
    base_task_project="deepfake-detection",
    base_task_name="Get pretrained model",
    parameter_override={"General/weights_url": "${pipeline.model_weights_url}"},
)

pipe.add_step(
    name="inference",
    parents=["load_model_weights"],
    base_task_project="deepfake-detection",
    base_task_name="Pretrained model inference",
)

# for debugging purposes use local jobs
pipe.start_locally()

# Starting the pipeline (in the background)
# pipe.start()

print("done")