from socket import *
from utils.Preprocess import Preprocess
from models.intent.IntentModel import IntentModel
from models.ner.NerModel import NerModel
import json

# Preprocess 객체 초기화
p = Preprocess(word2index_dic=r'C:\Users\skybr\OneDrive\Desktop\chatbot_yeji\train_tools\dict\chatbot_dict.bin',
               userdic=r'C:\Users\skybr\OneDrive\Desktop\chatbot_yeji\utils\user_dic.tsv')

# IntentModel 초기화
intent = IntentModel(model_name=r'C:\Users\skybr\OneDrive\Desktop\chatbot_yeji\models\intent\intent_model.h5', proprocess=p)

# NerModel 초기화 
ner = NerModel(model_name=r'C:\Users\skybr\OneDrive\Desktop\chatbot_yeji\models\ner\ner_model.h5', proprocess=p)

# 소켓 설정
HOST = '' 
PORT = 9999  
ADDR = (HOST, PORT)
BUFSIZE = 1024

server_socket = socket(AF_INET, SOCK_STREAM)
server_socket.bind(ADDR)
server_socket.listen(5) # 최대 5개의 클라이언트 연결 요청을 동시에 처리

print('연결을 기다리는 중입니다')

while True:
    client_socket, addr = server_socket.accept()
    print('연결이 되었습니다:', addr)

    # 클라이언트로부터의 요청을 기다림
    data = client_socket.recv(BUFSIZE)
    if data:
        received_text = data.decode('utf-8')
        print("받은 메시지:", received_text)

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

            # 시간 정보에 해당하는 단어를 추출 
            time_words= [word[0] for word, tag in zip(predicts, tags) if tag == 'B_DT']
            time = ' '.join(time_words)

        # 의도 예측
        predict = intent.predict_class(received_text)
        predict_label = intent.labels[predict]

        # 의도에 따른 응답 전송
        if predict_label == '인사':
            response = "안녕하세요. 무엇을 도와드릴까요?"
        elif predict_label == '욕설':
            response = "죄송하지만 욕설은 사용할 수 없습니다."
        elif predict_label == '주문':
            response = "네 알겠습니다. 목적지로 출발하겠습니다."
        elif predict_label == '예약':
            response = "네 알겠습니다. 예약이 완료되었습니다."
        else:
            response = "잘 이해하지 못했습니다. 다시한번 말씀해주세요."

         # JSON으로 응답 데이터 묶기
        response_data = {
            "intention": predict_label,
            "response": response,
            "destination": destination,
            "time": time
        }
        # JSON 형식으로 변환하여 클라이언트에게 전송
        response_json = json.dumps(response_data, ensure_ascii=False)
        client_socket.send(response_json.encode('utf-8'))

        client_socket.close()  # 클라이언트의 요청에만 응답 후 소켓 종료
    else:
        client_socket.close()  # 데이터가 없는 경우에도 소켓을 닫음

server_socket.close()