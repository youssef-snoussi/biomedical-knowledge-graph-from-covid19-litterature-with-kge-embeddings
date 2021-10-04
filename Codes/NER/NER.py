import warnings
warnings.filterwarnings('ignore')
import sys
import os
import pickle 
from tqdm import tqdm
import spacy 
import scispacy 
from scispacy.linking import EntityLinker 
import en_core_sci_scibert

path = 'cord19_sentences/cord19_sentences.pkl'

with open(path, "rb") as f:
    sentences = pickle.load(f)

# Creating NLP pipeline 

spacy.require_gpu()     # Run the model on GPU
nlp = en_core_sci_scibert.load()
nlp.add_pipe("merge_entities")
nlp.add_pipe("merge_noun_chunks")

print("Pipeline created successfully")

#Defining linker configuration, linker name can be either umls, hpo, mesh...
print("Adding scispacy UMLS linker, this may take some time...")
nlp.add_pipe("scispacy_linker", config = {
    'resolve_abbreviations': True,
    'linker_name': "umls"
})
print("UMLS linker added successfully to the pipeline")

# defining useful functions 

### Named Entity Extraction
def ner(sent):
  doc = nlp(sent)
  uml_ent = []
  for ent in doc.ents:
    cui, score = link_ent(ent)
    if cui != None:
      uml_ent.append({"ent_text":str(ent), 
                      "cui": cui, 
                      "score": score, 
                      "ent_start": ent.start_char, 
                      "ent_end": ent.end_char})
  return uml_ent


# linking entities to the UMLS, we're taking only the entities with score == 1
def link_ent(ent):

  uml_ent_cls = ent._.umls_ents
  entity = [ [e, s] for e, s in uml_ent_cls if s>=0.80 ]

  try : 
    return entity[0][0], entity[0][1]
  except Exception:
    return [None, None]


ner_sentences = []

for sent in tqdm(sentences):
  try:
    ents = ner(sent['sentence'])
  except Exception as err:
    print('error occured during processing: ', err)
    ents = []
  if len(ents) > 1: 
    ner_sentences.append({'sentence': sent, 'entities': ents })

# Saving the sentences and the named entities to a pickle file

with open("cord19_sentences/ner_sentences1.pkl", "wb") as f :
    pickle.dump(ner_sentences, f)

print("Entities extraction finished successfully!")

sys.exit()