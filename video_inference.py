import subprocess
subprocess.run('cd /mnt/datascience1/FT_LLaVA/LLaVA', shell=True)
from llava.model import LlavaLlamaForCausalLM
from llava.conversation import conv_templates, SeparatorStyle
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
from llava.utils import disable_torch_init
from transformers import AutoTokenizer, AutoModelForCausalLM
import streamlit as st
import cv2
import os
import tempfile
from io import StringIO
import numpy as np
import matplotlib.pyplot as plt
import shutil
from PIL import Image
import torch


import time
import requests
import torch
from PIL import Image
from io import BytesIO

from transformers import AutoTokenizer
from transformers import AutoProcessor, LlavaForConditionalGeneration
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch.nn as nn
import re
import json
import time

from langchain_core.documents.base import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

#3rd version to incorporate time selection

st.set_page_config(page_title="Video to Text Demo", layout="wide")

st.title("Delphi Demo: Skin Problem Diagnose Tool")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.write("This is a demo created by Esperanto Technologies.")

dirname = os.path.dirname(__file__)
device = 'cuda:0'
####################### Adding Descrptions to the streamlit page #######################
col1, col2 = st.columns([0.5,0.5], gap="medium")

option1 = col1.selectbox(
   'Please select your age group:',
    ('Age under 18','Age 18 to 29','Age 30 to 39','Age 40 to 49','Age 50 to 59','Age 60 to 69','Age 70 to 79','Age over 80','prefer not to say'),    
   index=None,                                         # 第一次加载的默认值
   placeholder="Select sex at birth...",
)
# col1.write('You Age Group is :', option1)
age = option1

option2 = col2.selectbox(
   'Sex at birth: Risk of diseases vary according to sex',
    ('Female','Male','Intersex / Non-binary','Prefer not to say'),    
   index=None,                                         # 第一次加载的默认值
   placeholder="Select sex at birth...",
)
# col2.write('You selected:', option2)
sex = option2

options1 = st.multiselect(
    "With which racial or ethnic groups do you identify? Mark all boxes that apply.",
['American Indian or Alaska Native','Asian','Black or African American',
'Hispanic, Latino, or Spanish Origin','Middle Eastern or North African',
'Native Hawaiian or Pacific Islander','White',
'Another race or ethnicity not listed','Prefer not to answer'])                # 多选的默认值
# col1.write('You selected:', options)
race =options1
# ['Black or African American', 'Middle Eastern or North African']

col1, col2 = st.columns([0.5,0.5], gap="medium")

option3 = col1.selectbox(
   'How does your skin react to sun exposure? Some skin types are more prone to skin diseases',
    ('Always burns, never tans','Usually burns, lightly tans',
'Sometimes burns, evenly tans','Rarely burns, tans well',
'Very rarely burns, easily tans','Never burns, always tans',
'None of the above'),    
   index=None,                                         # 第一次加载的默认值
   placeholder="Select your skin type...",
)
# col1.write('You Age Group is :', option1)
skin = option3

option4 = col2.selectbox(
   'For how long have you had this skin issue?',
    ('1 day','Less than 1 week',
'1-4 weeks','1-3 months',
'3-12 months','More than 1 year',
'More than 5 years','Since childhood',
'None of the above'),    
   index=None,                                         # 第一次加载的默认值
   placeholder="Select the duration...",
)
duration = option4

options2 = st.multiselect(
    "Describe how the affected skin area feels. Select all that apply.",
['Raised or bumpy','Flat','Rough or flaky','Filled with fluid'])                # 多选的默认值
# col1.write('You selected:', options)
texture =options2

options3 = st.multiselect(
    "Are you experiencing any of the following with your skin issue? Select all that apply.",
['Concerning in appearance','Bleeding',
'Increasing in size','Darkening','Itching',
'Burning','Pain','None of the above'])                # 多选的默认值
# col1.write('You selected:', options)
condition_symptoms =options3

other_symptoms = ['Black or African American', 'Middle Eastern or North African']
options4 = st.multiselect(
    "Do you have any of these symptoms? Select all that apply.",
['Fever','Chills','Fatigue','Joint pain',
'Mouth sores','Shortness of breath','None of the above'])                # 多选的默认值
# col1.write('You selected:', options)
other_symptoms =options4

####################### Set layout for Inference Part #######################
col1, col2 = st.columns([0.45,0.55], gap="medium")
global vidcap

if "disabled" not in st.session_state:
    st.session_state.disabled = False

#https://discuss.streamlit.io/t/are-there-any-ways-to-clear-file-uploader-values-without-using-streamlit-form/40903 
if "file_uploader" not in st.session_state:
    st.session_state['file_uploader'] = 0

def reset():
    st.session_state.user_input = ""
    st.session_state.user_widget = ""
    st.session_state.messages = []
    st.session_state.disabled = False

@st.cache_resource()
def load_model():
    # Load LLaVA
    model_path = '/mnt/datascience1/FT_LLaVA/skin_model'
    model = LlavaLlamaForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16
                ).to(device='cuda')
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device=device)
    image_processor = vision_tower.image_processor
    return model, tokenizer, image_processor

@st.cache_resource()
def load_model1():
    checkpoint = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer_m = AutoTokenizer.from_pretrained(checkpoint, use_fast=False, revision="main")
    model_m = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map="sequential",
        revision="main",
    )
    config_m = AutoConfig.from_pretrained(checkpoint)
    return tokenizer_m, model_m, config_m


@st.cache_resource()
def load_model2():
    tokenizer = AutoTokenizer.from_pretrained("epfl-llm/meditron-7b",token = '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')
    model = AutoModelForCausalLM.from_pretrained("epfl-llm/meditron-7b",token = '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').to(device=device)
    return tokenizer, model

@st.cache_resource()
def load_embed():
    embedding_model = "sentence-transformers/all-MiniLM-l6-v2"
    embeddings = HuggingFaceEmbeddings(
                            model_name=embedding_model,
                            model_kwargs={'device': device},
                            )
    return embeddings

    # Not sure if we want to get the input features seperately and then feed into LLaVA
# We refer to LLaVA github and wrote the following code
def caption_image(image, prompt, temp):
    # force_cudnn_initialization()
    # image = Image.open(image_file).convert('RGB')
        
    disable_torch_init()
    conv_mode = "llava_v0"
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
    inp = f"{roles[0]}: {prompt}"
    inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    raw_prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(raw_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    start = time.time()
    with torch.inference_mode():
       output_ids = model.generate(input_ids, images=image_tensor, do_sample=True, temperature=temp,
                                   max_new_tokens=100, use_cache=True, stopping_criteria=[stopping_criteria])
    # outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    outputs = tokenizer.decode(output_ids[0,:]).strip()
    conv.messages[-1][-1] = outputs
    output = outputs.rsplit('</s>', 1)[0]
    text_generate_time = time.time() - start
    return output, text_generate_time

    '''
    model_id = '/mnt/datascience1/FT_LLaVA/skin_model'# "llava-hf/llava-1.5-13b-hf"
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True, 
    ).to(0)
    processor = AutoProcessor.from_pretrained(model_id)

    # Load Mistral
    
    checkpoint = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer_m = AutoTokenizer.from_pretrained(checkpoint, use_fast=False, revision="main")
    model_m = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map="sequential",
        revision="main",
    )
    config_m = AutoConfig.from_pretrained(checkpoint)
    return model, processor# , tokenizer_m, model_m, config_m'''


####################### Main #############################
# model, processor, tokenizer_m, model_m, config_m = load_model()
model, tokenizer, image_processor = load_model()
tokenizer_m, model_m, config_m = load_model1()
# tokenizer_t, model_t = load_model2()
embeddings = load_embed()

# Upload videos and choose the range of clip to do inference with
with col1:
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if (uploaded_file != None):
        image = Image.open(uploaded_file)
        col1.write("The image of skin: ")
        col1.image(image)
        # cv2.imwrite("_.jpg", cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        # upload = Image.open("_.jpg")
        # st.button("Reset Parameters",on_click=reset)

# Inference Main Part
with col2:
    if (uploaded_file != None):
        # For the demo, choose to deal with 2x2 only, which is often the best layout
        des = 'Here are some basic information about the patient: '
        if str(age) != 'AGE_UNKNOWN':
            des = des + 'The patient is within the age group of ' + str(age)
        if str(sex) == 'MALE' or str(sex) == 'FEMALE':
            des = des + '. The patient is a ' + str(sex)
        if str(skin) != 'None of the above':
            des = des + ". The patient's fitzpatrick skin type is " + str(skin) + '. '
        if 'Prefer not to answer' not in race:
            r = ''
            for i in range(len(race)):
                if i != 0:
                    r = r + ' and ' + str(race[i])
                else: r = r + str(race[i])
            des = des + 'The patient is ' + r + '. '

        t = ''
        for i in range(len(texture)):
            if i != 0:
                t = t + ' and ' + str(texture[i])
            else: t = t + str(texture[i])
        des = des + " The texture of the skin problem is " + t + '. '

        if 'None of the above' not in condition_symptoms:
            c = ''
            for i in range(len(condition_symptoms)):
                if i != 0:
                    c = c + ' and ' + str(condition_symptoms[i])
                else: c = c + str(condition_symptoms[i])
            des = des + 'The skin problem comes with ' + c + '. '

        if 'None of the above' not in other_symptoms:
            o = ''
            for i in range(len(other_symptoms)):
                if i != 0:
                    o = o + ' and ' + str(other_symptoms[i])
                else: o = o + str(other_symptoms[i])
            des = des + 'The skin problem comes with ' + o + '. '
        
        if str(duration) != 'None of the above':
            des = des + "The patient's skin problem has been " + str(duration) + '. '
    
        # Question = st.text_input('Question: ', 'Please diagnose what skin condition is most likely in the picture.')
        # user_question = re.sub(r'(?i)video', 'image', des + Question)
    
        # Also make LLaVA pay special attention to the question user made to Mistral
        # prompt_system = 'This is an image of a possible skin problem. Diagnose the problem as a dermatologist. Your answer should provide enough information to understand the context and the action taking place. Then, if possible, give elements to answer the following question specifically. '
        # prompt = f"USER: <image>\n{prompt_system}\nQUESTION: {user_question}\nASSISTANT:"

        # Prepare inputs for LLaVA and save outputs as msg
        # raw_image = upload.convert('RGB')
        # user_question = 'Please diagonose what skin problem is in the picture.'#  Give one precise answer.'
        # user_question = Question
        # image_path = '/mnt/datascience1/FT_LLaVA/skin_condition/skin/-57713172034409208.png'
        # msg, t= caption_image(image, user_question, temp = 0.2)# "_.jpg"
        
        # inputs = processor(prompt, raw_image, return_tensors='pt').to(0, torch.float16)
        # output_l = model.generate(**inputs, max_new_tokens=200, do_sample=False, temperature=0.1)
        # msg = processor.decode(output_l[0][:], skip_special_tokens=True)
                
        # Prepare inputs for Mistral, provide msg as context
        # prompt_system_m = "Above is the description of a video, done frame by frame. Using this context and your reasoning, answer the previous question only. DO NOT MENTION the collages, the images, or the description: act as if you were analyzing the video itself. You may assume that this sequence constitutes a coherent whole and discard information that seems inaccurate or weird. You may use your knowledge if needed. Do not explain your reasoning, be confident."
        # Prompt_m = f'[INST]Video description: {msg}\nQuestion: {user_question}{prompt_system_m}\n[/INST]'

        # Based on execution device, either use Pytorch (CUDA) or ONNX (ETSOC-1)
        # input_ids = tokenizer_m(Prompt_m, return_tensors='pt')['input_ids'].to(device='cuda')
        # output = model_m.generate(input_ids=input_ids, max_new_tokens=300)#, do_sample = True, temperature = 0.1)
        # output = tokenizer_m.decode(output[0])[len(Prompt_m)+4:-4]

        # Show Mistral Answer in streamlit 
        # st.session_state.messages.append({"role": "assistant", "content": msg[3:]})
        # st.chat_message("assistant").write(msg[3:])
def reset():
    st.session_state.user_input = ""
    st.session_state.user_widget = ""
    st.session_state.messages = []

st.subheader("**Discuss with the skin experts**") 
if (uploaded_file != None):
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    Question = 'Please diagnose what skin condition is most likely in the picture.'
    Question = st.chat_input('Please diagnose what skin condition is most likely in the picture.')
    Description_pip = st.toggle('Show messages retrieved by RAG')
    if Question:
        st.chat_message('user').markdown(Question)
        if 'diagnose' in Question or 'skin' in Question or 'picture' in Question or 'image' in Question:
            user_question = re.sub(r'(?i)video', 'image', des + 'Please diagnose what skin condition is most likely in the picture.')
            msg, t = caption_image(image, user_question, temp = 0.2)# "_.jpg"
            st.chat_message("assistant").markdown(msg[3:])
            st.session_state.messages.append({'role': 'user', 'content': Question})
            st.session_state.messages.append({"role": "assistant", "content": msg[3:]})

        else:
            user_question = re.sub(r'(?i)video', 'image', des + 'Please diagnose what skin condition is most likely in the picture.')
            msg, t = caption_image(image, user_question, temp = 0.05)# "_.jpg"
            text = Question # msg.replace('condition', 'problem')
            
            with open('/mnt/datascience1/FT_LLaVA/skin_condition/skin_Terms1.json') as file:
                data = json.load(file)
    
            documents = []
            for item in data:
                article = item['article']
                metadata = item.get('metadata', {})
                document = Document(page_content=article, metadata=metadata)
                documents.append(document)
            
            text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n"," "], chunk_size=1024, chunk_overlap=100)
            documents = text_splitter.split_documents(documents)
            vector_db = FAISS.from_documents(documents, embeddings)
            retriever = vector_db.as_retriever(search_kwargs={"k": 3, 'score_threshold':0.85})
            relevant_docs = retriever.get_relevant_documents(Question)
            context = ''
            for i in relevant_docs:
                context = context + i.page_content + '  \n' + 'Source: ' + i.metadata['source'] +'; Link: ' + i.metadata['url'] + '  \n'
        
            if Description_pip:
                st.write(context)

            prompt_system_m = "Above is the diagnose of a skin problem based on the picture taken. Using this context and your reasoning, answer the previous question only. You may use your knowledge if needed. Do not explain your reasoning, be confident."
            Prompt_m = f'[INST]Skin problem Diagonose: {msg}\nSkin problem Context: {context}\nQuestion: {Question}{prompt_system_m}\n[/INST]'
            input_ids = tokenizer_m(Prompt_m, return_tensors='pt')['input_ids'].to(device='cuda')
            output = model_m.generate(input_ids=input_ids, max_new_tokens=1024)#, do_sample = True, temperature = 0.1)
            output = tokenizer_m.decode(output[0])[len(Prompt_m)+5:-4]
            
            # Test out meditron
            # input_ids = tokenizer_t(Question, return_tensors='pt')['input_ids'].to(device='cuda')
            # output = model_t.generate(input_ids=input_ids, max_new_tokens=1024)#, do_sample = True, temperature = 0.1)
            # output = tokenizer_t.decode(output[0])[len(Question)+4:]

            st.chat_message("assistant").markdown(output)
            st.session_state.messages.append({'role': 'user', 'content': Question})
            st.session_state.messages.append({"role": "assistant", "content": output})

st.button("Reset Parameters",on_click=reset)