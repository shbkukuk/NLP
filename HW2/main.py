from helper import Preprocessing
from ngrams import Ngrams

# If you run first time, you shuold uncomments this lines 
#preprocessing = Preprocessing(dump='HW2/archive/wiki_00')
#preprocessing.creata_train_test()

unigrams = Ngrams(1)
unigrams.read_files()
unigrams.calculate_Ngrams()
unigrams.create_count_table()
unigrams.gt_smoothing()
unigrams.calculate_test_perplexity()
unigrams.generate_random()

print("####################################")

bigrams = Ngrams(2)
bigrams.read_files()
bigrams.calculate_Ngrams()
bigrams.create_count_table()
bigrams.gt_smoothing()
bigrams.calculate_test_perplexity()
bigrams.generate_random()

print("####################################")

digrams = Ngrams(3)
digrams.read_files()
digrams.calculate_Ngrams()
digrams.create_count_table()
digrams.gt_smoothing()
digrams.calculate_test_perplexity()
digrams.generate_random()


print("####################################")
