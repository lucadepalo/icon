from experta import *

class Questions(KnowledgeEngine):

    def __init__(self, symptom_map, if_not_matched, get_treatments, get_details):
        self.symptom_map = symptom_map
        self.if_not_matched = if_not_matched
        self.get_details = get_details
        self.get_treatments = get_treatments
        KnowledgeEngine.__init__(self)

    @DefFacts()
    def _initial_action(self):
        print("")
        print("Questo software aiuta l'utente a capire se soffre di malattie cardiache oppure semplicemente di ansia/ipocondria")
        print("")
        print("Ti capita di provare uno o più dei seguenti sintomi?")
        print("Rispondi si oppure no")
        print("")
        yield Fact(action="find_disease")


    #taking various input from user
    @Rule(Fact(action="find_disease"), NOT(Fact(pressione_petto=W())), salience=21)
    def symptom_0(self):
        print("Ti capita a volte di avvertire un senso di pressione sul petto? ")
        c=input()
        while c!='si' and c!='no':
            print("inserisci un valore corretto ")
            c=input()
        self.declare(Fact(pressione_petto=c))

    @Rule(Fact(action="find_disease"), NOT(Fact(sedentarieta=W())), salience=20)
    def symptom_1(self):
        print("Hai uno stile di vita sedentario? ")
        c = input()
        while c != 'si' and c != 'no':
            print("inserisci un valore corretto ")
            c = input()
        self.declare(Fact(sedentarieta=c))

    @Rule(Fact(action="find_disease"), NOT(Fact(fumatore=W())), salience=19)
    def symptom_2(self):
        print("fumi? ")
        c = input()
        while c != 'si' and c != 'no':
            print("inserisci un valore corretto ")
            c = input()
        self.declare(Fact(fumatore=c))

    @Rule(Fact(action="find_disease"), NOT(Fact(dispnea=W())), salience=18)
    def symptom_3(self):
        print("Ti capita di avere il fiatone dopo un'attività fisica leggera (ad es. salire le scale)? ")
        c = input()
        while c != 'si' and c != 'no':
            print("inserisci un valore corretto ")
            c = input()
        self.declare(Fact(dispnea=c))

    @Rule(Fact(action="find_disease"), NOT(Fact(genetica_cardio=W())), salience=17)
    def symptom_4(self):
        print("Hai casi di cardiopatie in famiglia? ")
        c = input()
        while c != 'si' and c != 'no':
            print("inserisci un valore corretto ")
            c = input()
        self.declare(Fact(genetica_cardio=c)

    @Rule(Fact(action="find_disease"), NOT(Fact(genetica_psico=W())), salience=16)
    def symptom_5(self):
        print("Hai casi di ansia, depressione o panico in famiglia? ")
        c = input()
        while c != 'si' and c != 'no':
            print("inserisci un valore corretto ")
            c = input()
        self.declare(Fact(genetica_psico=c))

    @Rule(Fact(action="find_disease"), NOT(Fact(nausea=W())), salience=15)
    def symptom_6(self):
        print("Ti capita di avvertire nausea in situazioni di stress? ")
        c = input()
        while c != 'si' and c != 'no':
            print("inserisci un valore corretto ")
            c = input()
        self.declare(Fact(nausea=c))

    @Rule(Fact(action="find_disease"), NOT(Fact(stress=W())), salience=14)
    def symptom_7(self):
        print("Hai vissuto situazioni stressanti nell'ultimo mese? ")
        c = input()
        while c != 'si' and c != 'no':
            print("inserisci un valore corretto ")
            c = input()
        self.declare(Fact(stress=c))

    @Rule(Fact(action="find_disease"), NOT(Fact(stanchezza=W())), salience=13)
    def symptom_8(self):
        print("Avverti spesso stanchezza anche a riposo? ")
        c = input()
        while c != 'si' and c != 'no':
            print("inserisci un valore corretto ")
            c = input()
        self.declare(Fact(stanchezza=c))

    @Rule(Fact(action="find_disease"), NOT(Fact(insonnia=W())), salience=12)
    def symptom_9(self):
        print("Hai avuto difficoltà a dormire nell'ultimo mese? ")
        c = input()
        while c != 'si' and c != 'no':
            print("inserisci un valore corretto ")
            c = input()
        self.declare(Fact(insonnia=c))

    @Rule(Fact(action="find_disease"), NOT(Fact(gonfiore=W())), salience=11)
    def symptom_10(self):
        print("Hai le gambe gonfie a volte? ")
        c = input()
        while c != 'si' and c != 'no':
            print("inserisci un valore corretto ")
            c = input()
        self.declare(Fact(gonfiore=c))

    @Rule(Fact(action="find_disease"), NOT(Fact(palpitazioni=W())), salience=10)
    def symptom_11(self):
        print("Ti capita di avvertire palpitazioni nel petto? ")
        c = input()
        while c != 'si' and c != 'no':
            print("inserisci un valore corretto ")
            c = input()
        self.declare(Fact(palpitazioni=c))

    @Rule(Fact(action="find_disease"), NOT(Fact(sovrappeso=W())), salience=9)
    def symptom_12(self):
        print("Sei sovrappeso? ")
        c = input()
        while c != 'si' and c != 'no':
            print("inserisci un valore corretto ")
            c = input()
        self.declare(Fact(sovrappeso=c))

    @Rule(Fact(action="find_disease"), NOT(Fact(diabete=W())), salience=8)
    def symptom_13(self):
        print("Soffri di diabete? ")
        c = input()
        while c != 'si' and c != 'no':
            print("inserisci un valore corretto ")
            c = input()
        self.declare(Fact(diabete=c))

    @Rule(Fact(action="find_disease"), NOT(Fact(dieta=W())), salience=7)
    def symptom_14(self):
        print("Hai una dieta poco equilibrata (molti grassi, molti zuccheri, molte carni rosse, pochi vegetali)? ")
        c = input()
        while c != 'si' and c != 'no':
            print("inserisci un valore corretto ")
            c = input()
        self.declare(Fact(dieta=c))

    @Rule(Fact(action="find_disease"), NOT(Fact(sudore=W())), salience=6)
    def symptom_15(self):
        print("Soffri di vampate di calore e/o sudorazione fredda? ")
        c = input()
        while c != 'si' and c != 'no':
            print("inserisci un valore corretto ")
            c = input()
        self.declare(Fact(sudore=c))

    @Rule(Fact(action="find_disease"), NOT(Fact(angina=W())), salience=5)
    def symptom_16(self):
        print("Avverti a volte dolore alla schiena, al braccio sinistro o che si irraggia sotto il mento? ")
        c = input()
        while c != 'si' and c != 'no':
            print("inserisci un valore corretto ")
            c = input()
        self.declare(Fact(angina=c))

    @Rule(Fact(action="find_disease"), NOT(Fact(ruminazione=W())), salience=4)
    def symptom_17(self):
        print("Trovi che sia difficile concentrarti su un'attività a causa dei tuoi pensieri? ")
        c = input()
        while c != 'si' and c != 'no':
            print("inserisci un valore corretto ")
            c = input()
        self.declare(Fact(ruminazione=c))

    @Rule(Fact(action="find_disease"), NOT(Fact(tremori=W())), salience=3)
    def symptom_18(self):
        print("Ti capita di avere tremori e formicolii? ")
        c = input()
        while c != 'si' and c != 'no':
            print("inserisci un valore corretto ")
            c = input()
        self.declare(Fact(tremori=c))

    @Rule(Fact(action="find_disease"), NOT(Fact(negativita=W())), salience=2)
    def symptom_19(self):
        print("Tendi a vedere il lato negativo delle cose? ")
        c = input()
        while c != 'si' and c != 'no':
            print("inserisci un valore corretto ")
            c = input()
        self.declare(Fact(negativita=c))

    @Rule(Fact(action="find_disease"), NOT(Fact(ipocondria=W())), salience=1)
    def symptom_20(self):
        print("In genere ritieni di essere particolarmente preoccupata/o e impressionabile riguardo alla salute? ")
        c = input()
        while c != 'si' and c != 'no':
            print("inserisci un valore corretto ")
            c = input()
        self.declare(Fact(ipocondria=c))

    #different rules checking for each disease match
    @Rule(
        Fact(action="find_disease"),
        Fact(pressione_petto="no"),
        Fact(sedentarieta="si"),
        Fact(fumatore="si"),
        Fact(dispnea="si"),
        Fact(genetica_cardio="si"),
        Fact(genetica_psico="no"),
        Fact(nausea="no"),
        Fact(stress="no"),
        Fact(stanchezza="si"),
        Fact(insonnia="no"),
        Fact(gonfiore="si"),
        Fact(palpitazioni="si"),
        Fact(sovrappeso="si"),
        Fact(diabete="si"),
        Fact(dieta="si"),
        Fact(sudore="no"),
        Fact(angina="si"),
        Fact(ruminazione="no"),
        Fact(tremori="no"),
        Fact(negativita="no"),
        Fact(ipocondria="no"),
    )
    def disease_0(self):
        self.declare(Fact(disease="Cardiopatia"))

    @Rule(
        Fact(action="find_disease"),
        Fact(pressione_petto="si"),
        Fact(sedentarieta="no"),
        Fact(fumatore="no"),
        Fact(dispnea="no"),
        Fact(genetica_cardio="no"),
        Fact(genetica_psico="si"),
        Fact(nausea="si"),
        Fact(stress="si"),
        Fact(stanchezza="no"),
        Fact(insonnia="si"),
        Fact(gonfiore="no"),
        Fact(palpitazioni="si"),
        Fact(sovrappeso="no"),
        Fact(diabete="no"),
        Fact(dieta="no"),
        Fact(sudore="si"),
        Fact(angina="no"),
        Fact(ruminazione="si"),
        Fact(tremori="si"),
        Fact(negativita="si"),
        Fact(ipocondria="si"),
    )
    def disease_1(self):
        self.declare(Fact(disease="Ansia"))

    #when the user's input doesn't match any disease in the knowledge base
    @Rule(Fact(action="find_disease"), Fact(disease=MATCH.disease), salience=-998)
    def disease(self, disease):
        print("")
        id_disease = disease
        disease_details = self.get_details(id_disease)
        treatments = self.get_treatments(id_disease)
        print("")
        print("I tuoi sintomi corrispondono a: %s\n" % (id_disease))
        print("Ecco una breve descrizione della patologia :\n")
        print(disease_details + "\n")
        print(
            "Ecco cosa fare quando si sospetta di avere questa patologia: \n"
        )
        print(treatments + "\n")

    @Rule(
        Fact(action="find_disease"),
        Fact(pressione_petto=MATCH.pressione_petto),
        Fact(sedentarieta=MATCH.sedentarieta),
        Fact(fumatore=MATCH.fumatore),
        Fact(dispnea=MATCH.dispnea),
        Fact(genetica_cardio=MATCH.genetica_cardio),
        Fact(genetica_psico=MATCH.genetica_psico),
        Fact(nausea=MATCH.nausea),
        Fact(stress=MATCH.stress),
        Fact(stanchezza=MATCH.stanchezza),
        Fact(insonnia=MATCH.insonnia),
        Fact(gonfiore=MATCH.gonfiore),
        Fact(palpitazioni=MATCH.palpitazioni),
        Fact(sovrappeso=MATCH.sovrappeso),
        Fact(diabete=MATCH.diabete),
        Fact(dieta=MATCH.dieta),
        Fact(sudore=MATCH.sudore),
        Fact(angina=MATCH.angina),
        Fact(ruminazione=MATCH.ruminazione),
        Fact(tremori=MATCH.tremori),
        Fact(negativita=MATCH.negativita),
        Fact(ipocondria=MATCH.ipocondria),
        NOT(Fact(disease=MATCH.disease)),
        salience=-999
    )
    def not_matched(
        self,
        pressione_petto,
        sedentarieta,
        fumatore,
        dispnea,
        genetica_cardio,
        genetica_psico,
        nausea,
        stress,
        stanchezza,
        insonnia,
        gonfiore,
        palpitazioni,
        sovrappeso,
        diabete,
        dieta,
        sudore,
        angina,
        ruminazione,
        tremori,
        negativita,
        ipocondria,
    ):
        print("\nI tuoi sintomi combaciano parzialmente con quelli della patologia. Ti invitiamo ad effettuare ulteriori analisi.")
        lis = [
            pressione_petto,
            sedentarieta,
            fumatore,
            dispnea,
            genetica_cardio,
            genetica_psico,
            nausea,
            stress,
            stanchezza,
            insonnia,
            gonfiore,
            palpitazioni,
            sovrappeso,
            diabete,
            dieta,
            sudore,
            angina,
            ruminazione,
            tremori,
            negativita,
            ipocondria,
        ]
        max_count = 0
        max_disease = ""
        for key, val in self.symptom_map.items():
            count = 0
            temp_list = eval(key)
            for j in range(0, len(lis)):
                if temp_list[j] == lis[j] and (lis[j] == "si" or lis[j] == "no"):
                    count = count + 1
            if count > max_count:
                max_count = count
                max_disease = val
        if max_disease != "":
            self.if_not_matched(max_disease)

