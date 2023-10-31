from clearml import InputModel, Task


args = {
    "weights_url": "https://github.com/selimsef/dfdc_deepfake_challenge/releases/download/0.0.1/final_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_36"
}

task = Task.init(
    project_name="deepfake-detection",
    task_name="Get pretrained model",
)

task.connect(args)

# Import an existing model 
input_model = InputModel.import_model(
   # Name for model in ClearML
   name='Pretrained EfficientNet model for deepfake classification',
   # Import the model using a URL
   weights_url=args["weights_url"],
   framework='PyTorch'
)

task.connect(input_model)