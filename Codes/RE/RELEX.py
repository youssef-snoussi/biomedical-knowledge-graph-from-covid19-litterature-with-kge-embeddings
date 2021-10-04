import warnings
warnings.filterwarnings('ignore')
import pickle
from tqdm import tqdm
import allennlp
import allennlp_models
from allennlp_models.pretrained import load_predictor

path = "cord19_sentences/ner_sentences1.pkl"

with open(path, 'rb') as f:
    sentences = pickle.load(f)

# defining our predictor 
print("downloading the open information extraction model")
predictor = load_predictor('structured-prediction-srl-bert', cuda_device=0)  # works on GPU
print('model loaded successfully!')
# defining useful functions 
##########################

def get_sent_triples(sent, ents):
    sent = sent["sentence"]
    predicted_ie = predictor.predict(sentence=sent)

    for p_i in predicted_ie['verbs']:
        relation = p_i['verb']
        oie_tag = p_i['tags']
        triples = openie(sent, oie_tag, ents, relation)

    if 'triples' in locals():
        return triples
    else : 
        return []

###########################
def openie(sent, oie_pred, entities, relation):

    triples = []
    arg0 = [ w for j, w in enumerate(sent.split()) if 'ARG0' in oie_pred[j]]
    arg1 = [ w for j, w in enumerate(sent.split()) if 'ARG1' in oie_pred[j]]
    subjects = []
    objects = []
    for ent in entities:
        if ent['ent_text'] in ' '.join(arg0):
            subjects.append(ent)
        if ent['ent_text'] in ' '.join(arg1):
            objects.append(ent)

    #subjects = [ u for e, u in entities if str(e) in ' '.join(arg0) ]
    #objects = [ u for e, u in entities if str(e) in ' '.join(arg1) ]
    
    if subjects and objects:
        for s in subjects:
            for o in objects:
                triples.append([s, relation, o])

    return triples

############

all_triples = []
sent_triples = []

for sents in tqdm(sentences):

    sent = sents["sentence"]
    ents = sents["entities"] 

    try:
        triples = get_sent_triples(sent, ents)
    except Exception:
        triples = []
    
    if triples:
        for t in triples:
            if t not in all_triples:
                all_triples.append(t)
                sent_triples.append([sent, t])

with open("sent_triples.pkl", "wb") as f:
    pickle.dump(sent_triples, f)
