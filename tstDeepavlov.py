from deeppavlov import build_model, configs

model_qa = build_model(configs.squad.squad_bert, download=True)
x = model_qa(['David noticed he had put on a lot of weight recently. \
            He examined his habits to try and figure out the reason.\
            He realized he\'d been eating too much fast food lately \
            He stopped going to burger places and started a vegetarian diet.\
           After a few weeks, he started to feel much better.'],
         ["Why did David put on a lot of weight recently?"])

print(x)

'''from deeppavlov import configs
from deeppavlov.core.common.file import read_json
from deeppavlov.core.commands.infer import build_model
import nltk

faq = build_model(configs.faq.tfidf_logreg_en_faq, download = True)
a = faq(["the taco is in china. The Rice is in India"],["Where is the taco?"])
print(a)'''