

# Learning Representations

## Dependencies 

* We require the followings<br>
fail loudly: https://github.com/steverab/failing-loudly<br>
torch-two-sample: https://github.com/josipd/torch-two-sample<br>
keras-resnet: https://github.com/broadinstitute/keras-resnet<br>



* We require the following dependencies:<br>
keras: https://github.com/keras-team/keras<br>
tensorflow: https://github.com/tensorflow/tensorflow<br>
pytorch: https://github.com/pytorch/pytorch<br>
sklearn: https://github.com/scikit-learn/scikit-learn<br>
matplotlib: https://github.com/matplotlib/matplotlib


## Configuration <br>
Dataset is provided by fail_loudly repo.

Things that can be configured <br>
* DR methods
* Sample size
* Number of rundom runs
* Significance level



## Run

To run Landmark with Fail loudly repo,
* First download or clone the Fail loudly repo and change those .py files with .py files python_failoudly_landmark folder. 
* Run `run_pipeline.py` is provided to run all confifurations once.
* Make sure to change `Sign Level` and `Random Runs` in `pipeline_landmark.py`
