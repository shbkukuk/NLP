import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from turkishnlp import detector


#nltk.download('stopwords')
#nltk.download('punkt')

class Preprocessing():
    def __init__(self,dump) -> None:
        self.dump = dump
        self.lines = None
        self.train_set = ''
        self.test_set =''
        self.train_set_path = 'HW2/inputs/train.txt'
        self.test_set_path = 'HW2/inputs/test.txt'
        

    def creata_train_test(self):
        
        self.get_words()
        x ,y = int((len(self.lines)*95)/100) ,int(len(self.lines) - ((len(self.lines)*95)/100))
        train_lines , test_lines = self.lines[0:x] , self.lines[y:]
        print("train size ", len(train_lines))
        print("test size", len(test_lines))
        with open(self.train_set_path,'w+',encoding='utf8') as train_file:
            for line in train_lines:
                train_file.write(line)
                train_file.write('\n')
            train_file.close()
        with open(self.test_set_path,'w+',encoding='utf8') as test_file:
            for line in test_lines:
                test_file.write(line)
                test_file.write('\n')
            test_file.close()
    
    def get_words(self):
        text = []
        with open(self.dump,'r',encoding='utf8') as file:
            lines = file.readlines()
            for line in lines:
                if line!="</doc>" and line!="\n" and line!='doc' and line!='':  
                    text.append(line.lower())

        b=[]
        for i in text :
            if i[0:3]!="<do":
                b.append(i.rstrip("\n"))
        get_lines = [line.translate(str.maketrans('', '', string.punctuation)) for line in b][:20000]
        self.lines = self.remove_stopwords_non_turkish(get_lines)
    
    def remove_stopwords_non_turkish(self,lines):
        obj = detector.TurkishNLP()
        obj.download()
        obj.create_word_set()
        sw =stopwords.words('turkish')
        sw2=["''",",",".","(",")","``",":",".",";","-","_","“","’","a","b","c","d","e","f","g","c","ç","d","e","f","g","h","ı","i","j","k","l","m","n","o","ö","p","r","s","ş","t","u","ü","v","y","z"]
        for s in sw2:
            sw.append(s)
        res = []
        for line in lines:
            words = word_tokenize(line)
            new_line = []
            for word in words:
                if not word in sw and obj.is_turkish(word) and not word =='doc':
                    new_line.append(word)
            if len(new_line) > 1:
                res.append(' '.join(new_line)) 
        return res
         

