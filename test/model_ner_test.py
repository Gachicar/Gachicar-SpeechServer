from utils.Preprocess import Preprocess
from models.ner.NerModel import NerModel

p = Preprocess(word2index_dic=r'C:\Users\skybr\OneDrive\Desktop\chatbot_yeji\train_tools\dict\chatbot_dict.bin',
               userdic=r'C:\Users\skybr\OneDrive\Desktop\chatbot_yeji\utils\user_dic.tsv')


ner = NerModel(model_name=r'C:\Users\skybr\OneDrive\Desktop\chatbot_yeji\models\ner\ner_model.h5', proprocess=p)
# query = '오늘 오전 13시 2분에 학교로 설정 하고 싶어요'
query = '안녕'
predicts = ner.predict(query)
tags = ner.predict_tags(query)
print(predicts)
print(tags)

# food_words = [word[0] for word, tag in zip(predicts, tags) if tag == 'B_FOOD']
# print("장소 관련 단어:", ', '.join(food_words))
