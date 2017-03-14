REVIEWS_FILE = 'train-text.txt'
LABELS_FILE = 'train-labels.txt'
TRAINING_TEXT_TARGET = 'train-text-set1.txt'
TRAINING_LABELS_TARGET = 'train-labels-set1.txt'
TEST_TEXT_TARGET = 'test-text-set1.txt'
TEST_LABELS_TARGET = 'test-labels-set1.txt'

NO_OF_REVIEWS = 1280
TRAINING_SPLIT = 0.75
DEV_TEST_SPLIT = 0.25

# to split training set (review text)
with open(REVIEWS_FILE, 'r') as src, open(TRAINING_TEXT_TARGET, 'w') as target1, open(TEST_TEXT_TARGET, 'w') as target2:
    train_split_count = int(NO_OF_REVIEWS*TRAINING_SPLIT)
    test_split_count = int(NO_OF_REVIEWS*DEV_TEST_SPLIT)
    
    training_part = [next(src) for x in range(train_split_count)]
    [target1.write(line) for line in training_part]
    
    test_part = [next(src) for x in range(test_split_count)]
    [target2.write(line) for line in test_part]
    

# to split training set (review labels)
with open(LABELS_FILE, 'r') as src, open(TRAINING_LABELS_TARGET, 'w') as target1, open(TEST_LABELS_TARGET, 'w') as target2:
    train_split_count = int(NO_OF_REVIEWS*TRAINING_SPLIT)
    test_split_count = int(NO_OF_REVIEWS*DEV_TEST_SPLIT)
    
    training_part = [next(src) for x in range(train_split_count)]
    [target1.write(line) for line in training_part]
    
    test_part = [next(src) for x in range(test_split_count)]
    [target2.write(line) for line in test_part]
