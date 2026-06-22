#-*- coding: utf-8 -*-
#PROGRAMMA 1

import sys
import nltk
import string
from nltk.corpus import wordnet 
from nltk.stem import WordNetLemmatizer


#1. Funzione che calcola il numero delle frasi e il numero dei token di ciascun corpus
def calcola_frasi_token(testo):
    frasi = nltk.tokenize.sent_tokenize(testo) #tokenizzazione del testo in frasi
    numero_frasi = len(frasi)
    token = nltk.tokenize.word_tokenize(testo) #tokenizzazione del testo in parole
    numero_token = len(token)
    return frasi, token, numero_frasi, numero_token


#2. Funzioni che calcolano la lunghezza media della frasi in token e la lunghezza media dei token, ad eccezione della punteggiatura, in caratteri
def lunghezza_media_frasi(token, frasi):
    lung_media_frasi = len(token) / len(frasi) #calcolo la lunghezza media delle frasi in token
    return lung_media_frasi

def lunghezza_media_token(token):
    caratteri = 0
    num_token = 0 #inizializzo a 0 i contatori per caratteri e num_token
    for t in token:
        if t not in string.punctuation: #escludo i segni di punteggiatura
            caratteri += len(t)
            num_token += 1            
            lung_media_token = caratteri / num_token #calcolo la lunghezza media dei token
    return lung_media_token


#3. Funzione che calcola il numero di hapax tra i primi 500, 1000, 3000 token e nell'intero corpus
def num_hapax(token):
    valori = [500, 1000, 3000, len(token)] 
    risultati = [] #creo una lista per i valori da considerare e una lista per memorizzare i risultati
    for valore in valori: 
        conta_hapax = 0 
        for t in token[0:valore]:
            if token.count(t) == 1: #se il token appare solo una volta è un hapax e lo aggiunge alla lista
                conta_hapax += 1 
        risultati.append((valore, conta_hapax)) #il risultato viene aggiungo alla lista
    return risultati


#4. Funzioni che calcolano la dimensione del vocabolario e la ricchezza lessicale calcolata per porzioni incrementali di 200 token
def porzioni_testo(token, upper_bound = 200): #calcolo la dimensione del vocabolario e la TTR per una porzione di testo
    porzione_token = token[0:upper_bound] #seleziono la porzione di testo 
    set_vocabolario = list(set(porzione_token)) #calcolo la dimensione del vocabolario
    valore_ttr = TTR(porzione_token) #la TTR è calcolata richiamando la funzione
    return len(set_vocabolario), valore_ttr

def TTR(token): #funzione che calcola la TTR ovvero il rappporto tra parole tipo e la porzione di testo interessata
    parole_tipo = list(set(token))
    return len(parole_tipo) / len(token)

def dim_vocabolario_ttr(token, incremento = 200): #funzione che calcola la dimensione del vocabolario e la TTR per porzioni incrementali di 200 token
    step = incremento
    risultati = [] #creo una lista per memorizzare i risultati
    while step <= len(token): #ciclo per calcolare il vocabolario e la TTR per ogni porzione di testo con incremento di 200
        vocabolario, ttr = porzioni_testo(token, upper_bound = step)
        risultati.append((step, vocabolario, ttr)) #il risultato viene aggiungo alla lista
        step += incremento
    if step > len(token): #se supera la lunghezza totale del testo calcola il vocabolario e la TTR per l'intero corpus
        vocabolario, ttr = porzioni_testo(token, upper_bound = len(token))
        risultati.append ((step, vocabolario, ttr)) #il risultato viene aggiungo alla lista
    return risultati


#5. Funzione che calcola la dimensione dei lemmi distinti
def converti_pos(PoS): #funzione per convertire i tag POS di NLTK in quelli compatibili con WordNet
    if PoS.startswith("J"):
        return wordnet.ADJ
    elif PoS.startswith("V"):
        return wordnet.VERB
    elif PoS.startswith("N"):
        return wordnet.NOUN
    elif PoS.startswith("R"): 
        return wordnet.ADV 
    else:
        return wordnet.NOUN

def numeroLemmiDistinti(token):
    lemmatizzazione=WordNetLemmatizer() #inizializzo il lemmatizzatore
    lemmi = []
    post = nltk.tag.pos_tag(token) #applico il POS tagging ai token e assegno a ogni token un'etichetta grammaticale
    for token, pos in post: 
        pos_wordnet = converti_pos(pos) #converto l'etichetta POS di NLTK in un formato compatibile con WordNet
        lemma = lemmatizzazione.lemmatize(token, pos_wordnet) #lemmatizzo il token utilizzando l'etichetta POS convertita
        lemmi.append(lemma) #aggiungo il lemma alla lista
        vocabolario_lemmi = list(set(lemmi)) #creo un vocabolario di lemmi distinti
    return len(vocabolario_lemmi)


#Funzione principale che richiama le rispettive funzioni e stampa gli elementi richiesti
def main(file1, file2):
    with open(file1, 'r', encoding='utf-8') as fileInput1: #apro e leggo il file1
        raw1 = fileInput1.read()    
    with open(file2, 'r', encoding='utf-8') as fileInput2: #apro e leggo il file2
        raw2 = fileInput2.read()     
    #calcolo delle frasi e dei token per ciascun file     
    frasi1, token1, numero_frasi1, numero_token1 = calcola_frasi_token(raw1)
    frasi2, token2, numero_frasi2, numero_token2 = calcola_frasi_token(raw2)

    print("Progetto realizzato da Valentina Cosenza \n")
        
    #Di seguito stampo tutti i risultati ottenuti 
    print("PROGRAMMA 1 \n\n") 
    
    print("1. NUMERO FRASI E TOKEN \n")
    
    print("File 1 - The most famous speeches of Queen Elizabeth II \n")
    print("Il file 1 è composto da", numero_frasi1, "frasi e da", numero_token1, "tokens. \n")
    
    print("\nFile 2 - The Picture of Dorian Grey \n")
    print("Il file 2 è composto da", numero_frasi2, "frasi e da", numero_token2, "tokens. \n\n")


    print("2. LUNGHEZZA MEDIA DELLE FRASI IN TOKEN E LUNGHEZZA MEDIA DEI TOKEN, AD ECCEZIONE DELLA PUNTEGGIATURA, IN CARATTERI \n")
    lung_media1 = lunghezza_media_frasi(token1, frasi1)
    lung_media2 = lunghezza_media_frasi(token2, frasi2)
    lung_media_token1 = lunghezza_media_token(token1)
    lung_media_token2 = lunghezza_media_token(token2)

    print("File 1 - The most famous speeches of Queen Elizabeth II \n")
    print("Il file 1 ha una lunghezza media di frase di", lung_media1, "tokens e ha una lunghezza media di token di", lung_media_token1, "caratteri. \n")
   
    print("\nFile 2 - The Picture of Dorian Grey \n")
    print("Il file 2 ha una lunghezza media di frase di", lung_media2, "tokens e ha una lunghezza media di token di", lung_media_token2, "caratteri. \n\n")
    

    print("3. NUMERO DI HAPAX TRA I PRIMI 500, 1000, 3000, TOKEN E NELL'INTERO CORPUS \n")
    hapaxFile1 = num_hapax(token1)
    hapaxFile2 = num_hapax(token2)
        
    print("File 1 - The most famous speeches of Queen Elizabeth II \n")
    for valore, conta_hapax in hapaxFile1:
        print(f"Il numero di hapax nei primi {valore} token del corpus è: {conta_hapax}. \n")
    
    print("\nFile 2 - The Picture of Dorian Grey \n")
    for valore, conta_hapax in hapaxFile2:
        print(f"Il numero di hapax nei primi {valore} token del corpus è: {conta_hapax}. \n")
    

    print("\n4. DIMENSIONE DEL VOCABOLARIO E RICCHEZZA LESSICALE CALCOLATA PER PORZIONI INCREMENTALI DI 200 TOKEN \n")
    vocabolarioTTR1 = dim_vocabolario_ttr(token1)
    vocabolarioTTR2 = dim_vocabolario_ttr(token2)

    print("File 1 - The most famous speeches of Queen Elizabeth II \n")
    for incremento, vocabolario, ttr in vocabolarioTTR1:
        print(f"La dimensione del vocabolario per i primi {incremento} token del corpus è di {vocabolario} parole tipo.")
        print(f"L'indice di ricchezza lessicale(TTR) per i primi {incremento} token del corpus è pari a {ttr}. \n")
    
    print("\nFile 2 - The Picture of Dorian Grey \n")
    for incremento, vocabolario, ttr in vocabolarioTTR2:
        print(f"La dimensione del vocabolario per i primi {incremento} token del corpus è di {vocabolario} parole tipo.")
        print(f"L'indice di ricchezza lessicale(TTR) per i primi {incremento} token del corpus è pari a {ttr}. \n")
    

    print("\n5. NUMERO DEI LEMMI DISTINTI \n")
    lemmi1 = numeroLemmiDistinti(token1)
    lemmi2 = numeroLemmiDistinti(token2)

    print("File 1 - The most famous speeches of Queen Elizabeth II \n")
    print("Il file 1 contiene", lemmi1, "lemmi distinti. \n")
    
    print("\nFile 2 - The Picture of Dorian Grey \n")
    print("Il file 2 contiene", lemmi2, "lemmi distinti.")

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])