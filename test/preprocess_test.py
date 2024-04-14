from utils.Preprocess import Preprocess

# sent = "3월 25일 12시 30분으로 예약해줘"
# p = Preprocess(userdic=r"C:\Users\skybr\OneDrive\Desktop\chatbot_yeji\utils\user_dic3.tsv")

# pos = p.pos(sent)
# ret = p.get_keywords(pos, without_tag=False)

# print(pos)
# print(ret)
from utils.Preprocess import Preprocess

sent = "3월 25일 12시 30분으로 예약해줘"
p = Preprocess(userdic=r"C:\Users\skybr\OneDrive\Desktop\chatbot_yeji\utils\user_dic.tsv")

pos = p.pos(sent)
ret = p.get_keywords(pos, without_tag=False)
date = None

for keyword, pos in ret:
    if pos == 'NNP':
        date = keyword.replace(" ", "")

print(date)
print(pos)
print(ret)
