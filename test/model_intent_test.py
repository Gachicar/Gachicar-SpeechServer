from utils.Preprocess import Preprocess
from models.intent.IntentModel import IntentModel

p = Preprocess(word2index_dic=r'C:\Users\skybr\OneDrive\Desktop\chatbot_yeji\train_tools\dict\chatbot_dict.bin',
               userdic=r'C:\Users\skybr\OneDrive\Desktop\chatbot_yeji\utils\user_dic.tsv')

intent = IntentModel(model_name=r'C:\Users\skybr\OneDrive\Desktop\chatbot_yeji\models\intent\intent_model.h5', proprocess=p)

query = "안녕"
predict = intent.predict_class(query)
predict_label = intent.labels[predict]
print("="*30)
print(query)
print("의도 예측 클래스 : ", predict)
print("의도 예측 레이블 : ", predict_label)


query = "시발"
predict = intent.predict_class(query)
predict_label = intent.labels[predict]
print("="*30)
print(query)
print("의도 예측 클래스 : ", predict)
print("의도 예측 레이블 : ", predict_label)


query = "집으로 설정해줘"
predict = intent.predict_class(query)
predict_label = intent.labels[predict]
print("="*30)
print(query)
print("의도 예측 클래스 : ", predict)
print("의도 예측 레이블 : ", predict_label)


query = "내일 5시 30분에 학교로 예약해줘"
predict = intent.predict_class(query)
predict_label = intent.labels[predict]
print("="*30)
print(query)
print("의도 예측 클래스 : ", predict)
print("의도 예측 레이블 : ", predict_label)


query = "내 남자친구는 너무 귀여워"
predict = intent.predict_class(query)
predict_label = intent.labels[predict]
print("="*30)
print(query)
print("의도 예측 클래스 : ", predict)
print("의도 예측 레이블 : ", predict_label)
