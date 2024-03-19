from utils.Preprocess import Preprocess

sent = "내일 오전 10시에 윤용익에게로 가줘"
p = Preprocess(userdic=r"C:\Users\skybr\OneDrive\Desktop\chatbot_yeji\utils\user_dic.tsv")

pos = p.pos(sent)
ret = p.get_keywords(pos, without_tag=False)
print(ret)
