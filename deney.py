from skmultiflow.data.file_stream import FileStream

from Goowe import Goowe


# Prepare the data stream
stream = FileStream('./datasets/sea_stream.csv')
stream.prepare_for_use()

num_features = stream.n_features
num_targets = stream.n_targets
print(stream.get_target_values())
num_classes = len(stream.get_target_values())

N_MAX_CLASSIFIERS = 15
CHUNK_SIZE = 500        # User-specified
WINDOW_SIZE = 100       # User-specified

# Initialize the ensemble
goowe = Goowe(n_max_components=N_MAX_CLASSIFIERS,
              chunk_size=CHUNK_SIZE,
              window_size=WINDOW_SIZE)
goowe.prepare_post_analysis_req(num_features, num_targets, num_classes)

# For the first chunk, there is no prediction.
for i in range(CHUNK_SIZE):
    cur = stream.next_sample()
    X, y = cur[0], cur[1]
    goowe.partial_fit(X, y)

# Now, for the remaining instances, do ITTT (Interleaved Test Then Train).
while(stream.has_more_samples()):
    cur = stream.next_sample()
    X, y = cur[0], cur[1]
    goowe.predict(X)                        # Test
    goowe.partial_fit(X, y)             # Then train
