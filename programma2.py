#-*- coding: utf-8 -*-
#PROGRAMMA 2

import sys
import math
import nltk


#Funzione per eseguire il POS tagging
def pos_tagger(token):
    pos_tagging = nltk.tag.pos_tag(token) #applico il POS tagging ai token
    return pos_tagging


#1. Di seguito due funzioni che danno come risultato una sequenza ordinata per frequenza decrescente, con relativa frequenza, di: 
#a. Funzione per calcolare i 10 PoS, i 10 bigrammi di PoS e i 10 trigrammi di PoS più frequenti 
def dieci_elementi_più_frequenti(token, n):
    coppie_tp = pos_tagger(token) #creo una lista per le coppie token e PoS ottenuta invocando la funzione precedente
    s_pos = [] #creo una lista per memorizzare le sole PoS
    for token, pos in coppie_tp:
        s_pos.append(pos) #inserisco la PoS nella lista
    if n != 1: 
        ngrammi = list(nltk.ngrams(s_pos, n)) 
        freq = nltk.FreqDist(ngrammi) #genero ngrammi di PoS e calcolo le frequenze degli ngrammi
    else:
        freq = nltk.FreqDist(s_pos) #calcolo le frequenze delle singole PoS
    return freq.most_common(10)

#b. Funzione per calcolare i 20 sostantivi, avverbi e aggettivi più frequenti 
def venti_elementi_più_frequenti(coppie_tp, lista_tag):
    coppie_tt = [] #creo una lista per memorizzare i token filtrati
    for token, pos in coppie_tp:
        for tag in lista_tag:
            if pos == tag: #se il PoS del token corrisponde a uno dei tag specificati
                coppie_tt.append(token) #aggiungo il token alla lista
    freq = nltk.FreqDist(coppie_tt) #calcola le frequenze dei token filtrati
    return freq.most_common(20)


#2. Funzione per calcolare i bigrammi composti da Aggettivo e Sostantivo:
def bigrammi_agg_sost(coppie_tp): #funzione per estrarre i bigrammi 
    bigrammi = list(nltk.bigrams(coppie_tp))
    lista_bigrammi = [] #lista per memorizzare i bigrammi Aggettivo-Sostantivo
    for bigramma in bigrammi:
        if bigramma [0][1] == 'JJ' and (bigramma [1][1] == 'NN' or bigramma [1][1] == 'NNP' or bigramma [1][1] == 'NNPS' or bigramma [1][1] == 'NNS'):
            lista_bigrammi.append(bigramma) #aggiungo il bigramma alla lista
    return lista_bigrammi

#a. Funzione per calcolare i 20 bigrammi più frequenti, con relativa frequenza
def venti_bigrammi_frequenti(bigrammi):
    freq = nltk.FreqDist(bigrammi) #calcolo la frequenza dei bigrammi
    return freq.most_common(20) #restituisco i 20 bigrammi più frequenti

#b. Funzione per calcolare i 20 bigrammi con probabilità condizionata massima, e relativo valore di probabilità
def venti_bigrammi_pcm(token, lista_bigrammi):
    vocabolario_bigrammi = list(set(lista_bigrammi)) #creo un vocabolario per i bigrammi unici
    lista_bigrammi_pcm = [] #lista per memorizzare i bigrammi con probabilità condizionata
    for bigramma in vocabolario_bigrammi:
        freq_bigrammi = vocabolario_bigrammi.count(bigramma) #frequenza bigramma
        freq_token_1 = token.count(bigramma[0][0]) #frequenza token del bigramma
        if freq_token_1 != 0:
            pcm = freq_bigrammi / freq_token_1 #calcolo la probabilità condizionata massima
            lista_bigrammi_pcm.append((bigramma, pcm)) #aggiugo il bigramma e la probabilità alla lista
    ord_bigrammi = sorted(lista_bigrammi_pcm, key = lambda x: x[1], reverse = True) #ordina per probabilità
    return ord_bigrammi[:20] 

#c. Funzione per calcolare i 20 bigrammi con forza associativa massima e relativa PMI
def venti_bigrammi_pmi(token, lista_bigrammi):
    vocabolario_bigrammi = list(set(lista_bigrammi))
    lista_bigrammi_mi = [] #lista per memorizzare i bigrammi con il loro valore PMI
    for bigramma in vocabolario_bigrammi:
        prob_bigramma = lista_bigrammi.count(bigramma) / len(token) #calcolo la probabilità del bigramma
        prob_token_1 = token.count(bigramma[0][0]) / len(token) #calcolo la probabilità del primo token del bigramma
        prob_token_2 = token.count(bigramma[1][0]) / len(token) #calcolo la probabilità del secondo token del bigramma
        if prob_token_1 != 0 and prob_token_2 != 0:
            MI = round(math.log(prob_bigramma / (prob_token_1 * prob_token_2), 2)) #calcolo la PMI 
        lista_bigrammi_mi.append((bigramma, MI)) #aggiungo il bigramma alla lista
    ord_bigrammi = sorted(lista_bigrammi_mi, key = lambda x: x[1], reverse = True) #ordino i bigrammi in base alla PMI in ordine decrescente
    return ord_bigrammi[:20]


#3. Funzione per calcolare le frasi con una lunghezza compresa tra i 10 e i 20 token, in cui almeno la metà dei token occorre almeno 2 volte nel corpus:
def frasi_dieci_venti_token(token, frasi):
    frasi_finali = [] #creo una lista per memorizzare le frasi che soddisfano la condizione
    for frase in frasi:
        frase_token = nltk.word_tokenize(frase) 
        lung_frase = len(frase_token) #tokenizzo la frase e calcolo la lunghezza della frase in token
        if 10 <= lung_frase <= 20: #controllo che lalunghezza sia tra i 10 e i 20 token
            conta_no_hapax = 0 #contatore per i token che occorrono più di una volta nel corpus
            for t in frase_token:
                if token.count(t) > 1: #verifico se il token appare più di una volta
                    conta_no_hapax += 1
            if conta_no_hapax >= lung_frase / 2: #verifico se almeno  la metà dei token soddisfano la condizione
                frasi_finali.append(frase) #aggiungo la frase alla lista       
    return frasi_finali
        
#a. Funzione per trovare la frase con la media della distribuzione di frequenza dei token più alta
def media_distr_frequenza_alta(token, frasi):
    max_media = 0  #inizializzo la media massima
    max_frase = "" #inizializzo la frase con la media più alta
    for frase in frasi:
        frase_token = nltk.word_tokenize(frase)
        lung = len(frase_token)
        freq = 0 #contatore della somma della frequenza dei token
        for t in frase_token:
            freq += token.count(t) #somma della frequenza di ciascun token
        media = round(freq / lung) #calcolo la media
        if media > max_media: #verifico se la media è la più alta trovata finora
            max_media = media
            max_frase = frase #aggiorno la frase con la media più alta
    return max_media, max_frase

#b. Funzione per trovare la frase con la media della distribuzione di frequenza dei token più bassa
def media_distr_frequenza_bassa(token, frasi, max_media):
    min_media = max_media #inizializzo la media minima con il valore massimo iniziale
    min_frase = "" #inizializzo la frase con la media più bassa
    for frase in frasi:
        frase_token = nltk.word_tokenize(frase)
        lung = len(frase_token)
        freq = freq = 0
        for t in frase_token:
            freq += token.count(t)
        media = round(freq / lung)
        if media < min_media: #verifico se la media è la più bassa trovata finora
            min_media = media
            min_frase = frase #aggiorno la frase con la media più bassa
    return min_media, min_frase

#c. Funzione per calcolare la probabilità condizionata di un trigramma dato un bigramma
def prob_cond(trigramma, bigramma, token):
    bigrammi = list(nltk.bigrams(token)) #creo la lista di bigrammi del corpus
    trigrammi = list(nltk.trigrams(token)) #creo la lista di trigrammi del corpus
    prob_cond = trigrammi.count(trigramma) / bigrammi.count(bigramma) #calcolo la probabilità condizionata 
    return prob_cond
#Funzione per trovare la frase con la probabilità più alta secondo un modello di Markov di ordine 2
def markov_2(token, frasi):
    frasi_filtrate = frasi_dieci_venti_token(token, frasi) #seleziono le frasi valide
    bigrammi = list(nltk.bigrams(token)) #creo la lista dei bigrammi del corpus
    max_prob = 0 #inizializzo la probabilità massima
    max_frase = "" #inizializzo la frase con probabilità più alta
    for frase in frasi_filtrate:
        frase_token = nltk.word_tokenize(frase)
        frase_trigrammi = list(nltk.trigrams(frase_token)) #creo la lista di trigrammi della frase
        frase_bigrammi = list(nltk.bigrams(frase_token)) #creo la lista di bigrammi della frase
        prob1 = token.count(frase_token[0]) / len(token) #probabilità del primo token
        prob2 = bigrammi.count(frase_bigrammi[0]) / token.count(frase_bigrammi[0][0]) #probabilità del primo bigramma
        prob_frase = prob1 * prob2 #probabilità iniziale della frase
        k = 0 #contatore per iterare sui trigrammi
        for trigramma in frase_trigrammi:
            if k < len(frase_bigrammi): #controllo per evitare errori di indice
                bigramma = frase_bigrammi[k] #estraggo il bigramma corrispondente
                prob_cong = prob_cond(trigramma, bigramma, token) #calcolo la probabilità condizionata
                prob_frase *= prob_cong #aggiorno la probabilità della frase
                k += 1
        if prob_frase > max_prob: #verifico se la probabilità è la più alta trovata finora
            max_prob = prob_frase
            max_frase = frase #aggiorno la frase con probabilità più alta
    return max_frase, max_prob


#4. Funzione per estrarre le 15 entità nominate del testo:
def quindici_NE(token_pos, classe_NE):
    albero_NE = nltk.ne_chunk(token_pos) #creo l'albero delle entità nominate
    lista_NE = [] #creo una lista per memorizzare le entità nominate
    for nodo in albero_NE:
        if hasattr(nodo, 'label') and nodo.label() == classe_NE: #controllo se il nodo è della classe specificata
            tipo_entità = nodo.label() #identifico il tipo di entità
            entità = " ".join([token for token, POS in nodo.leaves()]) #unisco i token per formare il nome completo
            lista_NE.append(entità)  #aggiungo l'entità alla lista
    freq_NE = nltk.FreqDist(lista_NE) #calcolo la distribuzione di frequenza
    return freq_NE.most_common(15)


#Funzione principale che richiama le rispettive funzioni e stampa gli elementi richiesti
def main(file):
    with open(file, 'r', encoding='utf-8') as fileInput: #apro e leggo il file
        raw = fileInput.read()        
    
    token_1 = nltk.word_tokenize(raw) #tokenizzo il testo in token
    frasi_1 = nltk.sent_tokenize(raw) #suddivido il testo in frasi
    token_pos_1 = pos_tagger(token_1) #assegno i POS tag ai token

    print("Progetto realizzato da Valentina Cosenza \n")    

    #Di seguito stampo tutti i risultati ottenuti 
    print("PROGRAMMA 2 \n") 
    print(f"Analizzo il file - {file} \n\n")    

    print("1. FREQUENZA ORDINATA PER FREQUENZA DECRESCENTE, CON RELATIVA FREQUENZA, DI \n")

    print("a. 10 POS, BIGRAMMI DI POS E TRIGRAMMI DI POS PIÚ FREQUENTI: \n")

    print("I 10 PoS più frequenti nel corpus sono: \n")  
    for pos in dieci_elementi_più_frequenti(token_1, 1):        
        print(f"{pos} \n")

    print("\nI 10 bigrammi di Pos più frequenti nel corpus sono: \n")
    for bigrammi in dieci_elementi_più_frequenti(token_1, 2):
        print(f"{bigrammi} \n")

    print("\nI 10 trigrammi di Pos più frequenti nel corpus sono: \n")
    for trigrammi in dieci_elementi_più_frequenti(token_1, 3):
        print(f"{trigrammi} \n")
    
    print("\nb. 20 SOSTANTIVI, AVVERBI E AGGETTIVI PIÚ FREQUENTI: \n")
    
    print("I 20 sostantivi più frequenti nel corpus sono: \n")
    for sostantivi in venti_elementi_più_frequenti(token_pos_1, ['NN', 'NNP', 'NNPS', 'NNS']):
        print(f"{sostantivi} \n")
    
    print("\nI 20 avverbi più frequenti nel corpus sono: \n")
    for avverbi in venti_elementi_più_frequenti(token_pos_1, ['RB']):
        print(f"{avverbi} \n")
    
    print("\nI 20 aggettivi più frequenti nel corpus sono: \n")
    for aggettivi in venti_elementi_più_frequenti(token_pos_1, ['JJ']):
        print(f"{aggettivi} \n")
    

    print("\n2. ESTRARRE I BIGRAMMI COMPOSTI DA AGGETTIVO - SOSTANTIVO E MOSTRARE: \n")

    print("a. I 20 PIÚ FREQUENTI CON RELATIVA FREQUENZA \n") 
    lista_bigrammi_1 = bigrammi_agg_sost(token_pos_1)   
    
    print("I 20 bigrammi Aggettivo-Sostantivo più frequenti nel corpus sono: \n")
    for bigramma, frequenza in venti_bigrammi_frequenti(lista_bigrammi_1):
        print(f"{bigramma} -> {frequenza} \n")
    
    print("\nb. I 20 CON PROBABILITÁ CONDIZIONATA MASSIMA E RELATIVO VALORE DI PROBABILITÁ \n")

    print("I 20 bigrammi Aggettivo-Sostantivo con probabilità condizionata massima e relativo valore di probabilità sono: \n")    
    for bigramma, prob_condizionata in venti_bigrammi_pcm(token_1, lista_bigrammi_1):
        print(f"{bigramma} -> {prob_condizionata} \n")

    print("\nc. I 20 CON FORZA ASSOCIATIVA MASSIMA E RELATIVA PMI \n")

    print("I 20 bigrammi Aggettivo-Sostantivo con forza associativa massima e relativo valore di PMI sono: \n")
    for bigramma , pmi in venti_bigrammi_pmi(token_1, lista_bigrammi_1):
        print(f"{bigramma} -> {pmi} \n")
    

    print("\n3. CONSIDERARE LE FRASI CON UNA LUNGHEZZA COMPRESA TRA I 10 E I 20 TOKEN, IN CUI ALMENO LA METÁ DEI TOKEN OCCORE ALMENO 2 VOLTE NEL CORPUS(NON È UN HAPAX), SI IDENTIFICHINO: \n")

    print("a. LA FRASE CON LA MEDIA DELLA DISTRIBUZIONE DI FREQUENZA DEI TOKEN PIÙ ALTA \n") 
    
    print("La frase con la media della distribuzione di frequenza dei token più alta è:")
    max_media, frase = media_distr_frequenza_alta(token_1, frasi_1)
    print(f"'{frase}' con una media di frequenza di '{max_media}' \n")

    print("\nb. LA FRASE CON LA MEDIA DELLA DISTRIBUZIONE DI FREQUENZA DEI TOKEN PIÙ BASSA \n")

    print("La frase con la media della distribuzione di frequenza dei token più bassa è:")
    media, frase = media_distr_frequenza_bassa(token_1, frasi_1, max_media)
    print(f"'{frase}' con una media di frequenza di '{media}' \n")
    
    print("\nc. LA FRASE CON LA PROBABILITÁ PIÙ ALTA SECONDO UN MODELLO DI MARKOV DI ORDINE 2 COSTRUITO A PARTIRE DAL CORPUS DI INPUT \n")

    print("La frase con la probabilità più alta secondo il modello di Markov di ordine 2 è:")
    frase_markov, prob_markov = markov_2(token_1, frasi_1)
    print(f"'{frase_markov}' con una probabilità di '{prob_markov}' \n")
    

    print("\n4. ESTATTE LE ENTITÁ NOMINATE DEL TESTO, IDENTIFICARE PER CIASCUNA CLASSE DI NE I 15 ELEMENTI PIÙ FREQUENTI, ORDINATI PER FREQUENZA DESCESCENTE E CON RELATIVA FREQUENZA: \n")

    print("Le 15 entità nominate 'PERSON' più frequenti sono:\n")
    for person in quindici_NE(token_pos_1, 'PERSON'):
        print (f"{person} \n")
    
    print("\nLe 15 entità nominate 'GPE' più frequenti sono:\n")
    for gpe in quindici_NE(token_pos_1, 'GPE'):
        print (f"{gpe} \n")
    
    print("\nLe 15 entità nominate 'ORGANIZATION' più frequenti sono:\n")
    for organization in quindici_NE(token_pos_1, 'ORGANIZATION'):
        print (f"{organization} \n")
    
              
if __name__ == '__main__':
    main(sys.argv[1])
