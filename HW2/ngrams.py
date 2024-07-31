import math
import random
from nltk.util import ngrams
from turkishnlp import detector


class Ngrams:
    def __init__(self,size) :
        self.train = 'HW2/inputs/train.txt'
        self.test = 'HW2/inputs/test.txt'
        self.output = f'HW2/outputs/testperplexity_{str(size)}.txt'
        self.random = f'HW2/outputs/random_{str(size)}.txt'
        self.turkishNLP = detector.TurkishNLP()
        self.size = size
        self.train_txt = None
        self.test_txt = None
        self.ngram = None
        self.ngram_size = 0
        self.good_table = None
        self.count_table = None
        self.N = 0
        self.obj = detector.TurkishNLP()

    def read_files(self):
        with open(self.train,'r', encoding='utf8') as file:
            self.train_txt = [line[:-1] for line in file.readlines()]
            file.close()

        with open(self.test,'r',encoding='utf8') as file:
            self.test_txt =  [line[:-1] for line in file.readlines()]
            file.close()

    def flatten_extend(self,lst):
        flat_list = []
        for row in lst:
            flat_list.extend(row)
        return flat_list
    
    def counter(self):
        words_count=dict()
        for i in range(self.ngram_size):
            for key in self.ngram[i]:
                if key in words_count:
                    words_count[key] += 1
                else :
                    words_count[key] = 1
        
        self.good_table = words_count


    def seperate_syllabels(self,sentence):
        
        arr =self.obj.syllabicate_sentence(sentence)
        res =[]
        for element in arr:
            for innerelement in element:
                res.append(innerelement)
            res.append(' ')
        return res
    
    def calculate_Ngrams(self):
        syllables = []
        for sentence in self.test_txt:
            res =self.obj.syllabicate_sentence(sentence)
            syllables.extend(self.flatten_extend(res))
            syllables.append(" ")

        self.ngram = list(ngrams(syllables,self.size))
        self.ngram_size = len(self.ngram)
        self.counter()
        self.N = sum(self.good_table.values())

    def create_count_table(self):
        self.count_table = dict()
        for i in self.good_table:
            if self.good_table[i] in self.count_table:
                self.count_table[self.good_table[i]] += 1
            else:
                self.count_table[self.good_table[i]] = 1

    def gt_smoothing(self):
        if 1 in self.count_table:
            n1 = self.count_table[1]
        else :
            n1 = 1
            self.count_table[1] = n1

        for i in self.good_table:
            c = self.good_table[i]

            c0 = n1 / self.N
            if c not in self.count_table:
                self.good_table[i] = c0
            elif (c + 1) not in self.count_table:
                nc1 = c0
                nc = self.count_table[c]

                res = (((c + 1) * nc1) / nc)
                self.good_table[i] = res
            else:
                nc1 = self.count_table[c + 1]
                nc = self.count_table[c]

                res = (((c + 1) * nc1) / nc)
                self.good_table[i] = res


    def chainWithMarkovAssumption(self, sentence):
        sylArr = self.seperate_syllabels(sentence)

        if len(sylArr) < self.size:
            return 0

        gramList = list(ngrams(sylArr, self.size))
        
        logSum = sum(
            math.log10(self.good_table.get(i, self.count_table[1]) / self.N)
            for i in gramList
        )

        return math.exp(logSum)
    
    def calculate_test_perplexity(self):
        output = []
        for sentence in self.test_txt:
            result = self.chainWithMarkovAssumption(sentence)
            if result != 0:
                root = 1 / result
                perp = math.pow(root, 1 / self.size)
                output.append(sentence + " " + str(perp) + " " + "\n")
        
        with open(self.output,'w+',encoding='utf8') as file:
            file.writelines(output)
            file.close

    def generate_random(self):
        for i in range(5):
            rand = random.randrange(0,self.ngram_size)
            max = [0 for _ in range(5)]
            index = [0 for _ in range(5)]
            for j in range(10):
                for ind in range(self.size):
                    probility = self.good_table[self.ngram[rand][ind]]
                    if  probility> max[0]:
                        max[0] = probility
                        index[0] = rand
                    elif probility> max[1]:
                        max[1] = probility
                        index[1] = rand
                    elif probility> max[2]:
                        max[2] = probility
                        index[2] = rand
                    elif probility> max[3]:
                        max[3] = probility
                        index[3] = rand
                    elif probility> max[4]:
                        max[4] = probility
                        index[4] = rand
                
            print("".join(self.ngram[index[i]][0] for i in range(5)))
        print(" ")
