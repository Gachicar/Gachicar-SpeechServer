from socket import *
from utils.Preprocess import Preprocess
from models.intent.IntentModel import IntentModel
from models.ner.NerModel import NerModel
import json

# Preprocess 객체 초기화
p = Preprocess(word2index_dic=r'C:\Users\skybr\OneDrive\Desktop\chatbot_yeji\train_tools\dict\chatbot_dict.bin',
               userdic=r'C:\Users\skybr\OneDrive\Desktop\chatbot_yeji\utils\user_dic3.tsv')

# IntentModel 초기화
intent = IntentModel(model_name=r'C:\Users\skybr\OneDrive\Desktop\chatbot_yeji\models\intent\intent_model.h5', proprocess=p)

# NerModel 초기화 
ner = NerModel(model_name=r'C:\Users\skybr\OneDrive\Desktop\chatbot_yeji\models\ner\ner_model.h5', proprocess=p)

# 데이터 입력 (테스트를 위해 직접 입력)
received_text = "일주일후 12시 30분으로 예약해줘"

# 예약 받기 위한 과정 
pos = p.pos(received_text)
ret = p.get_keywords(pos, without_tag=False)

# 태그에 해당하는 단어 추출
predicts = ner.predict(received_text)
tags = ner.predict_tags(received_text)
print(predicts)
print(tags)

# 예외 처리: predicts 또는 tags가 None일 경우
if predicts is None or tags is None:
    # print("예외: predicts 또는 tags가 None입니다.")
    response = ""
    destination  = ""
    time = ""
else :
    # 장소 정보에 해당하는 단어를 추출 
    destination_words= [word[0] for word, tag in zip(predicts, tags) if tag == 'B_FOOD']
    destination = ' '.join(destination_words)

    # # 시간 정보에 해당하는 단어를 추출 
    # time_words= [word[0] for word, tag in zip(predicts, tags) if tag == 'B_DT']
    # time = ' '.join(time_words)

# 의도 예측
predict = intent.predict_class(received_text)
predict_label = intent.labels[predict]

date = None
hour = None
minute = None

# 의도에 따른 응답 생성
if predict_label == '인사':
    response = "안녕하세요. 무엇을 도와드릴까요?"
elif predict_label == '욕설':
    response = "죄송하지만 욕설은 사용할 수 없습니다."
elif predict_label == '주문':
    response = "네 알겠습니다. 목적지로 출발하겠습니다."
elif predict_label == '예약':
    response = ""
    for keyword, pos in ret:
        if pos == 'NNP':
            date = keyword.replace(" ", "")
        elif pos == 'NR':
            if hour is None:
               hour = keyword
            else:
                minute = keyword
else:
    response = "잘 이해하지 못했습니다. 다시한번 말씀해주세요."

# JSON으로 응답 데이터 묶기
response_data = {
    "intention": predict_label,
    "response": response,
    "destination": destination,
    "date": date,  
    "hour" : hour,
    "minute" : minute
}
# JSON 형식으로 출력
response_json = json.dumps(response_data, ensure_ascii=False)
print(response_json)

# 병원으로 예약하고 싶어 -> 예약하실 날짜와 시간을 말씀해주세요

# 3월 25일 12시 30분으로 예약해줘 -> 네 알겠습니다. 몇시간 예약하시겠습니까? or 해당 시간은 이미 예약되어있습니다. 다른 시간을 말씀해주세요. 
# {"intention": "예약", "response": "", "destination": "", "date": "3월25일", "hour": "12시", "minute": "30분"}
# 2시간 예약해줘 -> 예약이 정상적으로 완료 되었습니다. 
# {"intention": "예약", "response": "", "destination": "", "date": null, "hour": "2시", "minute": null}

# 예약한 시간이 되면 -> 백엔드에서 "예약한 시간이 되어 학교로 출발하겠습니다." 라는 메세지 필요 