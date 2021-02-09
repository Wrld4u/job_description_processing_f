import string
import pymorphy2
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.util import ngrams 
from tqdm.auto import tqdm
import numpy as np
from deeppavlov.core.common.file import read_json
from deeppavlov import build_model, configs
from sklearn.metrics.pairwise import cosine_similarity
from operator import itemgetter
import re
import regex
from Levenshtein import distance
from deeppavlov.models.preprocessors.str_lower import str_lower
from deeppavlov.models.tokenizers.nltk_tokenizer import NLTKTokenizer
from deeppavlov.models.embedders.fasttext_embedder import FasttextEmbedder
import pandas as pd

embedder = FasttextEmbedder(load_path="./ft_native_300_ru_wiki_lenta_lemmatize.bin")
tokenizer_simple = NLTKTokenizer()
morph = pymorphy2.MorphAnalyzer()
tokenizer = RegexpTokenizer('\w+|[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]')


def prepare_dataset(text_list):
    pad_token = '<unk>'
    documents = []
    documents_tags = []
    for text in text_list:
        split_text = [token for token in tokenizer.tokenize(text)]
        lem_text = [morph.parse(token)[0].normal_form 
                    if morph.parse(token)[0].normal_form not in stopwords.words('russian') \
                    and token not in string.punctuation \
                    else pad_token \
                    for token in split_text]
        documents.append(lem_text)
        tag_text = [['_'.join([morph.parse(token)[0].normal_form,
                               str(morph.parse(token)[0].tag.POS)]) 
                      for token in split_text]
                    ]
        documents_tags.append(tag_text)

    documents_tags = [doc[0] for doc in documents_tags]
        
    return documents, documents_tags

def prepare_dataset_not_normal_form(text_list):
    documents = []
    for text in text_list:
        split_text = [token for token in tokenizer.tokenize(text)]
        documents.append(split_text)
    return documents

def generate_ngrams(documents, max_len=None):
    ngramms = []
    for seq in documents:
        seq_ngramms = []
        for i in range(1, max_len):
            seq_ngramms += [' '.join(trip) 
                            for trip in list(ngrams(seq, i)) 
                            if '<unk>' not in trip]

        ngramms += [seq_ngramms]
    return ngramms

def init_bert(bert_path):
    bert_config = read_json(configs.embedder.bert_embedder)
    bert_config['metadata']['variables']['BERT_PATH'] = bert_path

    bert_model = build_model(bert_config)

    return bert_model


def location_recognition(ner_model,text):
    location_tags = []
    for n,i in enumerate(ner_model([text])[1][0]):
        result = re.search(r'LOC', i)
        if result != None:
            #print(result.group(0),ner_model([text])[0][0][n])
            location_tags.append(ner_model([text])[0][0][n])    
    return location_tags


def get_nearest_terms(terms, terms_db, threshold=0.2):
    if len(terms_db) == 0:
        return term
    terms_dist = [(t,distance(term.lower(),t.lower())) for t in terms_db for term in terms]
    sort_terms = sorted(terms_dist, key=lambda x:x[1])
    top_token = sort_terms[0] 
    if type(threshold) is int and top_token[1] < threshold:
        return (top_token[0], top_token[1])
    elif type(threshold) is float and top_token[1]/len(top_token[0]) < threshold:
        return (top_token[0], top_token[1]/len(top_token[0]))
    else:
        return (terms, 1.)

def l2_norm(x):
    return np.sqrt(np.sum(x**2))

def div_norm(x):
    norm_value = l2_norm(x)
    if norm_value > 0:
        return x * ( 1.0 / norm_value)
    else:
        return x
    

def get_sent_fasttext_emb(text_string):
    tags = tokenizer_simple(str_lower([text_string]))
    tags_embs = embedder(tags)
    tags_embs_norm = [div_norm(e) for e in tags_embs[0]]
    arr = np.array(tags_embs_norm)
    sent_emb = arr.sum(axis=0)/len(tags[0])
    return sent_emb

def fuzzy_search(text_string, terms_db, treshold = 0.15, ngrams =3, prepare = 1, correction_typo = 0 ):
    if prepare == 1:
        documents, _ = prepare_dataset([text_string])
    else:
        documents = prepare_dataset_not_normal_form([text_string])
    documents_ngamms = generate_ngrams(documents, ngrams)
    res_list = []
    for i in documents_ngamms[0]:
        token, score = get_nearest_terms([i], terms_db , treshold)
        if all([score <= treshold, token not in res_list, correction_typo == 1]):
            res_list.append(token)
        elif all([score <= treshold, i not in res_list, correction_typo != 1]):
            res_list.append(i)
    res_list_tags = ['#'+'_'.join(t.lower().split(' ')) for t in res_list]                
    return res_list_tags

def embs_sim_search_best_ngrams(text_string, terms_db, treshold = 0.82):
    documents, _ = prepare_dataset([text_string])
    documents_ngamms = generate_ngrams(documents, 3)
    df_p_embs = [[i, get_sent_fasttext_emb(i)] for i in terms_db]
    temp = {}
    for i in documents_ngamms[0]:
        for p_embs in df_p_embs:
            sim = cosine_similarity([get_sent_fasttext_emb(i)],[p_embs[1]])[0][0]
            if sim > treshold:
                if len(i.split(' ')) in temp:
                    if temp[len(i.split(' '))][1] < sim:
                        temp[len(i.split(' '))] = [i,sim]
                else:
                    temp[len(i.split(' '))] = [i,sim]            
    prof_list = []
    for i in temp.values():
        prof_list.append('#'+'_'.join(i[0].split(' ')))   
    return prof_list


def embs_sim_search(text_string, terms_db, treshold = 0.82):
    documents, _ = prepare_dataset([text_string])
    documents_ngamms = generate_ngrams(documents, 3)
    df_p_embs = [[i, get_sent_fasttext_emb(i)] for i in terms_db]
    temp = {}
    for i in documents_ngamms[0]:
        for p_embs in df_p_embs:
            sim = cosine_similarity([get_sent_fasttext_emb(i)],[p_embs[1]])[0][0]
            if sim > treshold:
                if i in temp:
                    if temp[i] < sim:
                        temp[i] = sim
                else:
                    temp[i] = sim 
    prof_list = []
    for i in temp.keys():
        prof_list.append('#'+'_'.join(i.split(' ')))   
    return prof_list
    
    

def remove_some_punctuation(text):
    return regex.sub(r'[^\w\s\-\/\\\(\)\.\,]', "", text)
def remove_space(text):
    return regex.sub(r'\s+', "", text)
def remove_tel_numb(text):
    return regex.sub(r'\+?\d[\( -]?\d{3}[\) -]?\d{3}[ -]?\d{2}[ -]?\d{2}', "", text)
def preproc_to_salary(text):
    return remove_tel_numb(remove_space(remove_some_punctuation(text)))

def get_month_salary(text):
    temp_ = preproc_to_salary(text)
    pattern = re.compile(r'\d{2}\s?\,?\.?\d{3}')
    salary = re.findall(pattern, temp_)
    if salary:
        salary_tags = ['#'+ regex.sub(r'\.?\,?', "", s) for s in salary]
    else:
        pattern = re.compile(r'\d{2}т{1}')
        salary = re.findall(pattern, temp_)
        if salary:
            pattern = re.compile(r'\d{2}')
            salary_tags = ['#'+ re.findall(pattern, s)[0] + '000' for s in salary]
    return salary_tags

def get_hour_salary(text):
    temp_ = preproc_to_salary('n'+text+ 'n')
    ind_list = [m.start() for m in re.finditer('(?=\d{3}[^\d]{0,5}(час)+)', temp_)]
    salary_list = []
    if ind_list:
        salary_list = ['#'+str(temp_[ind:ind+3])+'_час' for ind in ind_list]
    if not salary_list:
        ind_list = [m.start() for m in re.finditer('(?=(час)+.{0,5}\d{3})', temp_)]
        if ind_list:
            pattern = re.compile(r'\d{3}')
            for ind in ind_list:
                sal = re.findall(pattern, temp_[ind:ind+10])
                if sal:
                    salary_list.append(str('#'+str(sal[0])+'_час'))
    return salary_list

def get_smena_salary(text):
    temp_ = preproc_to_salary('n'+text+ 'n')
    ind_list = [m.start() for m in re.finditer('(?=\d{4}[^\d]{0,5}(смен)+)', temp_)]
    salary_list = []
    if ind_list:
        salary_list = ['#'+str(temp_[ind:ind+4])+'_смена' for ind in ind_list]
    if not salary_list:
        ind_list = [m.start() for m in re.finditer('(?=(смен)+.{0,5}\d{4})', temp_)]
        if ind_list:
            pattern = re.compile(r'\d{4}')
            for ind in ind_list:
                sal = re.findall(pattern, temp_[ind:ind+10])
                if sal:
                    salary_list.append(str('#'+str(sal[0])+'_смена'))
    return salary_list


def get_job_type(text):
    job_type = 0
    if get_month_salary(text):
        job_type = 1
    if job_type == 0:
        job_type_tag = '#подработка'
    else:
        job_type_tag = '#постоянная_работа'        
    return job_type_tag

def init_prof_clean():
    df = pd.read_excel('./datasets/data_mos_prepare_prof.xlsx') # data.mos
    df_p = df[0]
    prof_list = [i.lower() for i in df_p]
    return prof_list

def init_prof():
    df = pd.read_excel('./datasets/data_professii.xlsx') # data.mos
    df_p = df['NAME']
    return df_p

def init_loc():
    df_mos = pd.read_excel('./datasets/data_mos_metro_xlsx.xlsx') # data.mos
    df_m = df_mos['Station']
    df_obl = pd.read_excel('./datasets/data_msk_obl.xlsx', header=None)
    df_c = df_obl[0]
    loc = df_c.append(df_m, ignore_index=True)
    return loc

def get_tags(text):
    tags = []
    df_prof = init_prof_clean()
    treshold_prof = 0.2
    tags_prof = fuzzy_search(text, df_prof, treshold_prof,2)
    if not tags_prof:
        df_prof = init_prof()
        treshold_prof_embs = 0.82
        tags_prof = embs_sim_search(text, df_prof.values, treshold_prof_embs)
    tags = [t for t in tags_prof if t not in tags]
    df_loc = init_loc()
    treshold_loc = 0.2
    tags_loc = fuzzy_search(text, df_loc.values, treshold_loc,3,0,1)
    tags.extend([t for t in tags_loc if t not in tags])
    tags_sal = get_month_salary(text)
    tags.extend([t for t in tags_sal if t not in tags])
    tags_sal = get_smena_salary(text)
    tags.extend([t for t in tags_sal if t not in tags])
    tags_sal = get_hour_salary(text)
    tags.extend([t for t in tags_sal if t not in tags])
    tags_type_job = get_job_type(text)
    tags.extend([tags_type_job])
    
    return tags
