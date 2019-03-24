from skmultiflow.data.file_stream import FileStream
from skmultiflow.data import SEAGenerator
from skmultiflow.trees import HoeffdingTree
from skmultiflow.evaluation import EvaluatePrequential

from Goowe import Goowe


# Prepare the data stream
stream = FileStream('./datasets/og/covtype.csv')

stream.prepare_for_use()

num_features = stream.n_features
num_targets = stream.n_targets
num_classes = stream.n_classes
target_values = stream.target_values

print("Dataset with num_features:{}, num_targets:{}, num_classes:{}".format(
      num_features, num_targets, num_classes))


N_MAX_CLASSIFIERS = num_classes
CHUNK_SIZE = 100        # User-specified
WINDOW_SIZE = 100       # User-specified

# Initialize the ensemble
goowe = Goowe(n_max_components=N_MAX_CLASSIFIERS,
              chunk_size=CHUNK_SIZE,
              window_size=WINDOW_SIZE)
goowe.prepare_post_analysis_req(num_features, num_targets,
                                num_classes, target_values, record=True)

ht = HoeffdingTree()

evaluator = EvaluatePrequential(max_samples=100000,
                                max_time=1000,
                                pretrain_size=CHUNK_SIZE,
                                batch_size=1,
                                n_wait=CHUNK_SIZE,
                                show_plot=True,
                                output_file="out.txt",
                                metrics=['accuracy', 'kappa'])

# evaluator.evaluate(stream=stream, model=goowe, model_names=['GOOWE'])
evaluator.evaluate(stream=stream, model=[goowe, ht], model_names=['GOOWE', 'HT'])
