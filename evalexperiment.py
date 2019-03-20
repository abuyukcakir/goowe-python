from skmultiflow.data.file_stream import FileStream
from skmultiflow.data import SEAGenerator
from skmultiflow.trees import HoeffdingTree
from skmultiflow.evaluation import EvaluatePrequential

from Goowe import Goowe


# Prepare the data stream
stream = FileStream('./datasets/sea_stream.csv')

stream.prepare_for_use()
num_features = stream.n_features
num_targets = stream.n_targets
num_classes = stream.n_classes


N_MAX_CLASSIFIERS = 15
CHUNK_SIZE = 500        # User-specified
WINDOW_SIZE = 100       # User-specified

# Initialize the ensemble
goowe = Goowe(n_max_components=N_MAX_CLASSIFIERS,
              chunk_size=CHUNK_SIZE,
              window_size=WINDOW_SIZE)
goowe.prepare_post_analysis_req(num_features, num_targets, num_classes)

ht = HoeffdingTree()

evaluator = EvaluatePrequential(max_samples=100000,
                                max_time=1000,
                                pretrain_size=CHUNK_SIZE,
                                batch_size=1,
                                n_wait=CHUNK_SIZE,
                                show_plot=True,
                                output_file="out.txt",
                                metrics=['accuracy', 'kappa'])

evaluator.evaluate(stream=stream, model=[goowe, ht], model_names=['GOOWE', 'HT'])
