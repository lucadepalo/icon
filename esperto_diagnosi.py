from domande import Questions
from ontologia import Ontology
diseases_list = []
diseases_symptoms = []
symptom_map = {}
d_desc_map = {}
d_treatment_map = {}

#loads the knowledge from .txt files into variables to allow the code to use it
def preprocess():
    #global diseases_list, diseases_symptoms, symptom_map, d_desc_map, d_treatment_map
    diseases = open("data/malattie.txt")
    diseases_t = diseases.read()
    diseases_list = diseases_t.split("\n")
    diseases.close()

    for disease in diseases_list:
        disease_s_file = open("data/" + disease + "_sintomi.txt")
        disease_s_data = disease_s_file.read()
        s_list = disease_s_data.split("\n")
        diseases_symptoms.append(s_list)
        symptom_map[str(s_list)] = disease
        disease_s_file.close()

        disease_s_file = open("data/" + disease + "_descrizione.txt")
        disease_s_data = disease_s_file.read()
        d_desc_map[disease] = disease_s_data
        disease_s_file.close()

        disease_s_file = open("data/" + disease + "_suggerimenti.txt")
        disease_s_data = disease_s_file.read()
        d_treatment_map[disease] = disease_s_data
        disease_s_file.close()


def identify_disease(*arguments):
    symptom_list = []
    for symptom in arguments:
        symptom_list.append(symptom)

    return symptom_map[str(symptom_list)]


def get_details(disease):
    return d_desc_map[disease]


def get_treatments(disease):
    return d_treatment_map[disease]


def if_not_matched(disease):
    print("")
    id_disease = disease
    disease_details = get_details(id_disease)
    treatments = get_treatments(id_disease)
    print("")
    print("I tuoi sintomi corrispondono a: %s\n" % (id_disease))
    print("Ecco una breve descrizione della patologia :\n")
    print(disease_details + "\n")
    print(
        "Ecco cosa fare quando si sospetta di avere questa patologia: \n"
    )
    print(treatments + "\n")

def main_ontology():
    ont = Ontology()
    ont.get_symptoms_descriptions()
    symptoms, keys_symptoms = ont.print_symptoms()
    while True:
        print("\nSeleziona il sintomo di cui vuoi conosere la descrizione, inserisci il numero del sintomo")
        symptom_number = int(input())

        while symptom_number not in symptoms.keys():
            print("\nAttenzione: numero errato. Seleziona il sintomo di cui vuoi conosere la descrizione, inserisci il numero del sintomo")
            symptom_number = int(input())

        print("Sintomo: %s, descrizione: %s" % (keys_symptoms[symptom_number], " ".join(symptoms[symptom_number])))
        print("\nVuoi leggere dettagli su un altro sintomo?\nRispondi con si o no")
        if input() == "no":
            exit()

#driver function
if __name__ == "__main__":
    preprocess()
    #creating class object
    engine = Questions(symptom_map, if_not_matched, get_treatments, get_details)
    #loop to keep running the code until user says no when asked for another diagnosis
    while 1:
        engine.reset()
        engine.run()
        print("Vuoi ripetere il test?\n Rispondi con si o no")
        if input() == "no":
            print("Vuoi visualizzare ulteriori informazioni sui sintomi presenti nelle domande appena lette?\n Rispondi con si o no")
            if input() == "si":
                main_ontology()
            else:
                exit()

