# Welcome to the War Soldier! This is your one-stop hub for everything PokeAttack related. 
# This is the project submission for team BaPU (Ballistic Pokemon Unaliver) - made by ee24b141 and ee24b156.
# We recommend that you put all models in the same directory as this file, there will be 4 files to store regarding models:
#     1) best.pt
#     2) electra
#     3) w2v.model
#     4) embeddings

# Place the prompt data in test_prompts_orders.py, in the format given. The testing data we used is already provided in it. 
# Change IMAGE_LOCATION to the directory with the test images 
IMAGE_LOCATION = "test_images" # Image dataset
ELECTRA_LOC = "./electra" # Fine-tuned model 

# ======================================================================================================================== 
# ⢀⢀⢀⢀⢀⢀⢀⢀⢀⢀⢀⢀⢀⢀⣠⠖⠹⡍⠉⠉⡗⢦
#⢀⢀⢀⢀⢀⢀⢀⢀⢀⢀⢀⢀⢀⢀⢰⢁⣠⠴⠿⠿⠿⢷⣤⣳
#⢀⢀⢀⠴⠶⠒⠲⢄⡀⢀⢀⢀⢀⢀⡿⡿⢀⢀⢀⡤⣄⢀⢀⠙
#⢀⢀⡼⢀⢀⢀⢀⢀⡇⢀⢀⢀⢀⢀⠇⡏⢀⠂⠉⢀⢀⠉⠉⢉⢀
#⢀⠐⣇⢀⣀⣀⢀⢀⡇⢀⢀⢀⢀⢻⠚⢉⡝⢲⡄⢀⢀⢰⠛⡍⢦⠃
#⢀⢀⢸⠁⢀⢀⠈⠉⠳⢀⢀⢀⢀⢀⢆⢈⣓⣤⣇⣀⣀⣸⡤⢃⡸
#⢀⢀⢸⠒⢲⠒⠂⢰⠃⢀⢀⢀⠤⣄⢀⡟⠢⠤⣤⣤⣤⣤⡬⠥⡀
#⢀⢀⠇⢀⠞⢀⢀⠸⡀⢀⠔⣅⢀⠈⠳⡉⠒⡤⠤⠤⠤⠤⣴⠒⡷⠁⠒⢦
#⢀⡜⡰⠁⢀⢀⢀⢀⢱⠊⢀⡜⢆⢀⢀⠈⢦⣇⣀⣀⣀⣀⣹⠎⢀⢀⢀⣼⠑⢄⢀⣀⢀
#⢀⡇⡇⢀⢀⢀⢀⢀⢈⡇⢀⡇⢀⠱⣄⠠⣤⣤⣤⢀⡀⢤⣤⡤⠄⢠⠎⢸⡀⢀⢙⣤⢀⢳
#⢰⠃⢱⢀⢀⢀⢀⢀⡸⢸⡀⢧⢀⢀⠈⠢⡀⢀⢀⢀⠁⢀⢀⢀⣰⠧⠄⠒⠛⠉⢀⡿⢀⠈⢇
#⢸⣆⠜⠓⠒⠒⠒⠊⠁⠈⡇⢀⠱⡀⢀⢀⠈⢩⢿⡿⠿⠿⢖⢺⢀⢀⢀⢀⣀⡠⠴⡇⢀⢀⠸⡄
#⠘⡏⢀⢀⢀⢀⢀⢀⢀⢸⠇⢀⡠⠊⠉⠉⠉⠁⢀⢀⢀⢀⢀⠙⠻⢽⢹⡏⢀⢀⢀⢷⢀⢀⢀⡇
#⢀⢇⢀⢀⢀⢀⢀⢀⢀⡸⡠⠊⢀⣠⠞⠉⠉⠉⠉⠉⠉⠉⠉⠉⠑⢼⡆⡇⢀⢀⢀⠸⢀⢀⢸
#⢀⠈⠢⣀⢀⢀⢀⣀⡴⠋⢀⢀⠘⣅⢀⢀⢀⢀⢀⢀⢀⢀⢀⢀⢀⢀⡇⢠⢀⢀⢀⣠⡇⢀⡆
#   __________________   ____  __._____________      __  _____ __________ 
#   \______   \_____  \ |    |/ _|\_   _____/  \    /  \/  _  \\______   \
#   |     ___//   |   \|      <   |    __)_\   \/\/   /  /_\  \|       _/
#   |    |   /    |    \    |  \  |        \\        /    |    \    |   \
#   |____|   \_______  /____|__ \/_______  / \__/\  /\____|__  /____|_  /
#                     \/        \/        \/       \/         \/       \/ 
# ========================================================================================================================

import functools, csv
import numpy as np
import pickle
from tqdm import tqdm
from peft import PeftModel
from ultralytics import YOLO
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor
import os, csv, torch, json, spacy, pickle, gensim
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ========================================================================================================================

def LoadModel(load_dir, base_model_name="google/electra-base-discriminator"):
    ''' Loading our transformer model  --> Fine tuned ELECTRA-Base 115M  '''
    tokenizer = AutoTokenizer.from_pretrained(load_dir)
    base_model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=2)
    model = PeftModel.from_pretrained(base_model, load_dir)
    model.eval()
    return model, tokenizer

def DetectKillCommand(model, tokenizer, text: str) -> bool:
    ''' Runs a command through a fine-tuned model to check command status (Kill or not)'''
    model.eval()
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits.cpu()
    pred_class = torch.argmax(logits, dim=-1).item()
    return bool(pred_class)

def DetectKillLoc(model, tokenizer, text: str, nlp_model, id=0) -> list[tuple[str, float]]:
    ''' Parses HQ prompt using NLP and sieving methods to determine the Kill Command '''
    thresholds = {"electra": (0.05, 0.0045, 0.0001), "ft_electra": (0.1, 0.05, 0.0001)}
    THRESHOLD, FIRST_SIEVE, SECOND_SIEVE = thresholds["electra"]


    doc = nlp_model(text)
    sentences = [sent.text for sent in doc.sents]
    inputs = tokenizer(sentences, return_tensors='pt', truncation=True, padding=True) # Batch inferencing
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits.cpu()
    probs = torch.softmax(logits, dim=1)
    
    remaining_sentences_with_prob = [] 
    kill_commands = [] 
    for sent, prob in zip(sentences, probs):
        kill_prob = prob[1].item()
        if sent ==  "Prioritize preservation of scientific instruments in the area.":
            continue
        elif sent == "Mark LZ candidates; do not land within visible ashfall.":
            continue
        if kill_prob >= THRESHOLD:
            kill_commands.append((sent, kill_prob))
        else:
            remaining_sentences_with_prob.append((sent, kill_prob))

    # First sieve -> First relaxation of rules triggered if none match exactly
    if len(kill_commands) == 0:
        for i in remaining_sentences_with_prob:
            if i[1] >= FIRST_SIEVE:
                kill_commands.append(i)        
    
    # Second sieve -> Maximum relaxation
    if len(kill_commands) == 0:
        for i in remaining_sentences_with_prob:
            if i[1] >= SECOND_SIEVE:
                kill_commands.append(i)
    
    return kill_commands

def PromptObtain(DATA, model_dir=ELECTRA_LOC):
    results = {}
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    model, tokenizer = LoadModel(model_dir)

    print("Tokenizer used is default, hence the error" + "\n"*50)
    pbar = tqdm(total=len(DATA))
    for i in DATA:
        id = i["image_id"]
        prompt = i["prompt"]
        sample = {} 
        kill_commands = DetectKillLoc(model=model, tokenizer=tokenizer, text=prompt, nlp_model=nlp, id=id)  # Takes model and tokenizer along with prompt to parse and returns list
        sample[prompt] = kill_commands
        results[id] = sample
        pbar.update(1)
        
    with open("pokemon_targets.json", "w") as file:
        json.dump(results, file, indent=4)
    return results

def EmbedText(w2v, text):
    words = text.lower().split()
    valid_words = [w2v[word] for word in words if word in w2v]
    return np.mean(valid_words, axis=0) if valid_words else np.zeros(w2v.vector_size)

def FindClosestPokemon(description, nlp, w2v, pokemon_embedding):
    ''' Get the pokemon that is being described by the sentence '''
    doc = nlp(description)

    direct_ref = 0
    pokemon = ["Pikachu", "Bulbasaur", "Charizard", "Mewtwo"]
    pokemon_lower = [i.lower() for i in pokemon]
    for word in description.split():
        if word in pokemon or word in pokemon_lower:
            direct_ref = 1
            return word.title(), {}, direct_ref

    sentence = description

    # word2vec implementation
    query_vec = EmbedText(w2v, sentence)
    similarities = {}
    for name, emb in pokemon_embedding.items():
        if np.linalg.norm(query_vec)==0 or np.linalg.norm(emb)==0:
            sim = 0
        else:
            sim = cosine_similarity([query_vec], [emb])[0][0]
        similarities[name] = sim
    if similarities == {'Pikachu': 0, 'Charizard': 0, 'Bulbasaur': 0, 'Mewtwo': 0}:
        return None, None, None
    # Find pokemon with max similarity score
    return max(similarities, key=similarities.get), similarities, direct_ref

def PromptPoint( results, w2v_store="w2v.model" ):
    ''' Takes in a dict of list of possible kill commands and returns the pokemon most likely referred to'''
    with open("embeddings","rb") as file:
        pokemon_embedding = pickle.load(file)
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    w2v = gensim.models.KeyedVectors.load(w2v_store)

    pbar = tqdm(total=len(results))
    final_output = {}
    diagnostic = {}
    for id in results:
        samples = results[id]
        possibles = []

        for prompt in samples:
            kill_command = samples[prompt][0][0]
            predicted_pokemon, sims, direct = FindClosestPokemon(kill_command, nlp, w2v, pokemon_embedding)
            
            if predicted_pokemon == None:
                continue
            if direct == 1:
                possibles = [[kill_command, predicted_pokemon, str(sims), direct]]
                break
            possibles.append( (kill_command, predicted_pokemon, str(sims), direct) )

        
        if len(possibles) == 1:
            final_output[id] = possibles[0][1]
            diagnostic[id] = possibles
        elif len(possibles) == 0:
            final_output[id] = None
            diagnostic[id] = possibles
        elif len(possibles) > 1:
            # old version was final_output[id] = possibles
            max_sims = 0
            pokemon = ""
            diagnostic[id] = possibles
            for element in possibles:
                predicted_pokemon, sims = element[1], list(element[2])
                if max(sims) > max_sims:
                    max_sims = max(sims)
                    pokemon = predicted_pokemon
            final_output[id] = pokemon
        pbar.update(1)
            
    with open("pokemon_diagnostic.json","w") as file:
        json.dump(diagnostic, file, indent=4)
    with open("pokemon_pick.json", "w") as file:
        json.dump(final_output, file, indent=4)

    return final_output

def PromptPoint( results, w2v_store="w2v.model" ):
    ''' Takes in a dict of list of possible kill commands and returns the pokemon most likely referred to'''
    with open("embeddings","rb") as file:
        pokemon_embedding = pickle.load(file)
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    w2v = gensim.models.KeyedVectors.load(w2v_store)

    pbar = tqdm(total=len(results))
    final_output = {}
    diagnostic = {}
    for id in results:
        samples = results[id]
        possibles = []

        for prompt in samples:
            kill_command = samples[prompt][0][0]
            predicted_pokemon, sims, direct = FindClosestPokemon(kill_command, nlp, w2v, pokemon_embedding)
            
            if predicted_pokemon == None:
                continue
            if direct == 1:
                possibles = [[kill_command, predicted_pokemon, str(sims), direct]]
                break
            possibles.append( (kill_command, predicted_pokemon, str(sims), direct) )

        
        if len(possibles) == 1:
            final_output[id] = possibles[0][1]
            diagnostic[id] = possibles
        elif len(possibles) == 0:
            final_output[id] = None
            diagnostic[id] = possibles
        elif len(possibles) > 1:
            # old version was final_output[id] = possibles
            max_sims = 0
            pokemon = ""
            diagnostic[id] = possibles
            for element in possibles:
                predicted_pokemon, sims = element[1], list(element[2])
                if max(sims) > max_sims:
                    max_sims = max(sims)
                    pokemon = predicted_pokemon
            final_output[id] = pokemon
        pbar.update(1)
            
    with open("pokemon_diagnostic.json","w") as file:
        json.dump(diagnostic, file, indent=4)
    with open("pokemon_pick.json", "w") as file:
        json.dump(final_output, file, indent=4)

    return final_output
@functools.cache
def CoordinateObtain(source_dir):
    ''' Takes a folder of images and returns a dictionary of Pokemon Bounding Boxes for each image 
        Inputs: source_dir -> Location of images '''

    model = YOLO("best.pt")
    results = model.predict(source=source_dir, conf=0.7, iou=0.3, imgsz=800, verbose=False, save=True)
    data = {}

    for result in results:
        image_name = os.path.basename(result.path)
        pokemon_details = {}
        for box in result.boxes:
            class_id = int(box.cls)
            class_name = model.names[class_id]
            confidence = float(box.conf)
            bounding_box = box.xyxy[0].tolist()
            if class_name not in pokemon_details.keys():
                pokemon_details[class_name] = []

            pokemon_details[class_name].append({
                'confidence': confidence,
                'bbox': bounding_box
            })
        # pokemon_details.sort(key=lambda x: x['confidence'], reverse=True)
        data[image_name] = pokemon_details

    with open('bbox_targets.json', 'w') as file:
        json.dump(data, file, indent=4)

    return data

def TargetObtain( scene_details, targets, threshold ):
    def MiddleShot( coordinate ):
        x = ( coordinate[0] + coordinate[2] ) / 2
        y = ( coordinate[1] + coordinate[3] ) / 2
        return [x, y]

    with open("output.csv", "w") as file:
        writer = csv.writer(file)
        header = ["image_id","points"]
        writer.writerow(header)

        for id in scene_details.keys():
            shots = []
            target_class = targets[id]
            if target_class == None:
                writer.writerow([id, str([])])
                continue
            if target_class.lower() not in scene_details[id].keys():
                writer.writerow([id, str([])])
                continue
            coordinates = scene_details[id][target_class.lower()]
            for coord in coordinates:
                confidence = coord["confidence"]
                if confidence < threshold:
                    continue
                shot = MiddleShot( coord['bbox'] )
                shots.append(shot)

            submit = '[]'
            for coord in shots:
                submit += str(coord) + ", "
            submit += ']'
            writer.writerow([id, str(shots)])
       
# ======================================================================================================================== 

from test_prompts_orders import DATA

commands = PromptObtain(DATA)
targets = PromptPoint(commands)
scene_details = CoordinateObtain(source_dir=IMAGE_LOCATION)
TargetObtain( scene_details, targets, threshold=0.72 )
