from deap import tools
from deap import algorithms
from deap import base
from deap import creator
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import ast

# JEEC Member = JEECM

# Values to change according to JEEC's edition

# TODO COORDINATION
N_DIAS_JEEC = 9 # TODO 9
NUM_MIN_ELEMENTS_PER_DEP = 2
MAX_SHIFTS_PER_WEEK = 23
MIN_SHIFTS_PER_WEEK = 5 # TODO 5
MAX_SHIFTS_PER_WEEK_VOLUNTEER = 13
MIN_SHIFTS_PER_WEEK_VOLUNTEER = 3 # TODO 3
N_SHIFTS_PER_DAY = 7
departments = ['WebDev', 'Speakers', 'Business', 'Logistics', 'Marketing', 'Volunteer']

n_shifts = N_DIAS_JEEC * N_SHIFTS_PER_DAY

# Generate pessoas.xlsx based on forms.xlsx
def use_forms():
    name = 'forms'
    read_file = pd.read_excel (name + ".xlsx")
    read_file.to_csv (name + ".csv", 
                    index = None,
                    header=True, encoding='utf-8')

    forms = pd.read_csv(name + ".csv", encoding='utf-8') 

    # TODO COORDINATION
    
    # Match with forms.xlsx
    dias = ['Assinala de acordo com a tua disponibilidade. (preenche com a tua disponibilidade máxima, e.g., when2meet) [Dia 1]', 
            'Assinala de acordo com a tua disponibilidade. (preenche com a tua disponibilidade máxima, e.g., when2meet) [Dia 2]']

    # Match with forms.xlsx
    turnoss = ['9h-10:30h', '10:30h-12h', '12h-13:30h', '13:30h-15h' '15h-16:30h', '16:30h-18h', '18h-19:30h']

    to_exc = []

    # Iterar por cada pessoa
    for index, row in forms.iterrows():
        disponibilidade = [0 for _ in range(n_shifts)]

        for i in range(len(dias)):
            for j in range(len(turnoss)):
                row[dias[i]] = str(row[dias[i]])
                # print(row[dias[i]].split(sep=', '))
                if turnoss[j] in list(row[dias[i]].split(sep=', ')):
                    disponibilidade[i * N_SHIFTS_PER_DAY + j] = 1
                
        to_exc.append({'Nome': row['Nome'], 'Equipa': row['Equipa'], 'Disponibilidade': disponibilidade})

    to_exc = pd.DataFrame(to_exc)
    to_exc.to_excel("pessoas.xlsx", 
                    index = None,
                    header=True)
    
# use_forms() # TODO DESCOMENTAR

name_p = 'pessoas'
read_file = pd.read_excel (name_p + ".xlsx")
read_file.to_csv (name_p + ".csv", 
                  index = None,
                  header=True, encoding='utf-8')

name_t = 'turnos'
read_file = pd.read_excel (name_t + ".xlsx")
read_file.to_csv (name_t + ".csv", 
                  index = None,
                  header=True, encoding='utf-8')

pessoas = pd.read_csv(name_p + '.csv', encoding='utf-8') 
turnos = pd.read_csv(name_t + '.csv', encoding='utf-8') 

pref_shift = {}
for index, row in turnos.iterrows():
    pref_shift[row['turnos']] = ast.literal_eval(row['preferencia'])
    
pref_shift['Extra'] = []

# print(turnos)
# print(pref_shift)

        
def eaSimpleWithElitism(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__):
    """This algorithm is similar to DEAP eaSimple() algorithm, with the modification that
    halloffame is used to implement an elitism mechanism. The individuals contained in the
    halloffame are directly injected into the next generation and are not subject to the
    genetic operators of selection, crossover and mutation.
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is None:
        raise ValueError("halloffame parameter must not be empty!")

    halloffame.update(population)
    hof_size = len(halloffame.items) if halloffame.items else 0

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):

        # Select the next generation individuals
        offspring = toolbox.select(population, len(population) - hof_size)

        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # add the best back to population:
        offspring.extend(halloffame.items)

        # Update the hall of fame with the generated individuals
        halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook

class JEECMSchedulingProblem:
    """This class encapsulates the JEECM Scheduling problem
    """

    def __init__(self, hardConstraintPenalty):
        """
        :param hardConstraintPenalty: the penalty factor for a hard-constraint violation
        """
        self.hardConstraintPenalty = hardConstraintPenalty

        # list of JEECMs:
        self.JEECMs = pessoas['Nome']
        self.JEECMs_equipas = pessoas['Equipa']

        # JEECMs' respective shift preferences - morning, evening, night:
        self.shiftPreference = pessoas['Disponibilidade']

        # min and max number of JEECMs allowed for each shift - morning, evening, night:        
        self.shiftMin = []
        self.shiftMax = []

        for v, row in turnos.iterrows():
            dec = list(eval(row['decorre']))
            if v == 0:
                for i in range(len(dec)):
                    self.shiftMin.append(dec[i] * int(row['Num Pessoas']))
                    self.shiftMax.append(dec[i] * int(row['Num Pessoas']))
            else:
                for i in range(len(dec)):
                    self.shiftMin[i] += dec[i] * int(row['Num Pessoas'])
                    self.shiftMax[i] += dec[i] * int(row['Num Pessoas'])
                    
            self.shiftMax[i] += 10

        # max shifts per week allowed for each JEECM
        self.maxShiftsPerWeek = MAX_SHIFTS_PER_WEEK
        self.minShiftsPerWeek = MIN_SHIFTS_PER_WEEK
        self.maxShiftsPerWeekVol = MAX_SHIFTS_PER_WEEK_VOLUNTEER
        self.minShiftsPerWeekVol = MIN_SHIFTS_PER_WEEK_VOLUNTEER
        
    def __len__(self):
        """
        :return: the number of shifts in the schedule
        """
        return len(self.JEECMs) * n_shifts


    def getCost(self, schedule):
        """
        Calculates the total cost of the various violations in the given schedule
        ...
        :param schedule: a list of binary values describing the given schedule
        :return: the calculated cost
        """

        if len(schedule) != self.__len__():
            raise ValueError("size of schedule list should be equal to ", self.__len__())

        # convert entire schedule into a dictionary with a separate schedule for each JEECM:
        JEECMShiftsDict = self.getJEECMShifts(schedule)

        # count the various violations:
        consecutiveShiftViolations = self.countConsecutiveShiftViolations(JEECMShiftsDict)
        shiftsPerWeekViolations = self.countShiftsPerWeekViolations(JEECMShiftsDict)[1]
        JEECMsPerShiftViolations = self.countJEECMsPerShiftViolations(JEECMShiftsDict)[1]
        shiftPreferenceViolations = self.countShiftPreferenceViolations(JEECMShiftsDict)
        lessthan2perDep = self.countlessthan2perDep(JEECMShiftsDict)
        
        # calculate the cost of the violations:
        hardContstraintViolations = shiftPreferenceViolations + shiftsPerWeekViolations*10 + JEECMsPerShiftViolations + lessthan2perDep * 0.01
        softContstraintViolations =  - consecutiveShiftViolations*0.001

        return self.hardConstraintPenalty * hardContstraintViolations + softContstraintViolations

    def getJEECMShifts(self, schedule):
        """
        Converts the entire schedule into a dictionary with a separate schedule for each JEECM
        :param schedule: a list of binary values describing the given schedule
        :return: a dictionary with each JEECM as a key and the corresponding shifts as the value
        """
        shiftsPerJEECM = self.__len__() // len(self.JEECMs)
        JEECMShiftsDict = {}
        shiftIndex = 0

        for JEECM in self.JEECMs:
            JEECMShiftsDict[JEECM] = schedule[shiftIndex:shiftIndex + shiftsPerJEECM]
            shiftIndex += shiftsPerJEECM

        return JEECMShiftsDict

    def countConsecutiveShiftViolations(self, JEECMShiftsDict):
        """
        Counts the consecutive shift violations in the schedule
        :param JEECMShiftsDict: a dictionary with a separate schedule for each JEECM
        :return: count of violations found
        """
        violations = 0
        # iterate over the shifts of each JEECM:
        
        for JEECMShifts in JEECMShiftsDict.values():
            # look for two cosecutive '1's:
            kk = 0
            for shift1, shift2 in zip(JEECMShifts, JEECMShifts[1:]):
                if shift1 == 1 and shift2 == 1 and (kk + 1) % N_SHIFTS_PER_DAY != 0:
                    violations += 1
                kk += 1
                
        return violations

    def countShiftsPerWeekViolations(self, JEECMShiftsDict):
        """
        Counts the max-shifts-per-week violations in the schedule
        :param JEECMShiftsDict: a dictionary with a separate schedule for each JEECM
        :return: count of violations found
        """
        violations = 0
        weeklyShiftsList = []
        # iterate over the shifts of each JEECM:
            
        for member in JEECMShiftsDict:
            JEECMShifts = JEECMShiftsDict[member]
            # name = member
            # print(len(shifts))
            
            for i in range(len(self.JEECMs)):
                if self.JEECMs[i] == member:
                    equipa = self.JEECMs_equipas[i]
                    
            # iterate over the shifts of each weeks:
            for i in range(0, n_shifts, n_shifts):
                # count all the '1's over the week:
                weeklyShifts = sum(JEECMShifts[i:i + n_shifts])
                weeklyShiftsList.append(weeklyShifts)
                if weeklyShifts > self.maxShiftsPerWeek and equipa != 'Volunteer':
                    violations += weeklyShifts - self.maxShiftsPerWeek
                elif weeklyShifts < self.minShiftsPerWeek and equipa != 'Volunteer':
                    violations += self.minShiftsPerWeek - weeklyShifts
                elif weeklyShifts > self.maxShiftsPerWeekVol and equipa == 'Volunteer':
                    violations += weeklyShifts - self.maxShiftsPerWeekVol
                elif weeklyShifts < self.minShiftsPerWeekVol and equipa == 'Volunteer':
                    violations += self.minShiftsPerWeekVol - weeklyShifts

        return weeklyShiftsList, violations

    def countJEECMsPerShiftViolations(self, JEECMShiftsDict):
        """
        Counts the number-of-JEECMs-per-shift violations in the schedule
        :param JEECMShiftsDict: a dictionary with a separate schedule for each JEECM
        :return: count of violations found
        """
        # sum the shifts over all JEECMs:
        totalPerShiftList = [sum(shift) for shift in zip(*JEECMShiftsDict.values())]

        violations = 0
        # iterate over all shifts and count violations:
        for shiftIndex, numOfJEECMs in enumerate(totalPerShiftList):
            dailyShiftIndex = shiftIndex % n_shifts  # -> 0, 1, or 2 for the 3 shifts per day
            if (numOfJEECMs > self.shiftMax[dailyShiftIndex]):
                violations += numOfJEECMs - self.shiftMax[dailyShiftIndex]
            elif (numOfJEECMs < self.shiftMin[dailyShiftIndex]):
                violations += self.shiftMin[dailyShiftIndex] - numOfJEECMs

        return totalPerShiftList, violations

    def countShiftPreferenceViolations(self, JEECMShiftsDict):
        """
        Counts the JEECM-preferences violations in the schedule
        :param JEECMShiftsDict: a dictionary with a separate schedule for each JEECM
        :return: count of violations found
        """
        violations = 0
        for JEECMIndex, shiftPreference in enumerate(self.shiftPreference):
            # duplicate the shift-preference over the days of the period
            # preference = shiftPreference * (self.shiftsPerWeek // n_shifts)
            preference = shiftPreference
            # iterate over the shifts and compare to preferences:
            shifts = JEECMShiftsDict[self.JEECMs[JEECMIndex]]
            for pref, shift in zip(preference, shifts):
                if pref == 0 and shift == 1:
                    violations += 1

        return violations
    
    def countlessthan2perDep(self, JEECMShiftsDict):
        violations = 0
        
        people_per_shift = [[] for _ in range(n_shifts)]
        
        counters = {dep: 0 for dep in departments}
        # print(counters)
        
        for member in JEECMShiftsDict:
            shifts = JEECMShiftsDict[member]
            # name = member
            # print(len(shifts))
            for i in range(len(shifts)):
                if shifts[i] == 1:
                    people_per_shift[i].append(member)
        
        for i in range(len(people_per_shift)):
            if i >= N_SHIFTS_PER_DAY * 2 and i < n_shifts - N_SHIFTS_PER_DAY * 2: # nao contar com montagem/desm
                people_list = people_per_shift[i]
                for j in range(len(people_list)):
                    people = people_list[j]
                    # print(people)
                    # Find index of person and return team
                    # Funciona se nao houver nomes repetidos
                    for i in range(len(self.JEECMs)):
                        if self.JEECMs[i] == people:
                            equipa = self.JEECMs_equipas[i]
                    
                    counters[equipa] += 1
                
                for dep in departments:
                    if dep != 'Volunteer':
                        # if (counters[dep] < NUM_MIN_ELEMENTS_PER_DEP and dep != 'Marketing') or (counters[dep] < 1 and dep == 'Marketing'):
                        if counters[dep] < NUM_MIN_ELEMENTS_PER_DEP:
                            violations += 1
                    
                    #penalise a lot of volunteers
                    # if (counters[dep] > MAX_SHIFTS_PER_WEEK_VOLUNTEER and dep == 'Volunteer'):
                    #     violations += 1
                    counters[dep] = 0           

        return violations
    
    def countConsecutiveShiftViolations2(self, people_shifts):
        violations = 0
        for person_shifts in people_shifts:
            for i in range(1, n_shifts):
                if person_shifts[i-1] == person_shifts[i] and i % N_SHIFTS_PER_DAY != 0: # Não conta com turnos fim dia->inicio dia seguinte
                    violations+=1
        return violations

    def countShiftsPerWeekViolations2(self,final_to_computions):
        violations = 0
        list = []
        for person_shifts in final_to_computions:
            list.append(len(person_shifts))
            if len(person_shifts) > MAX_SHIFTS_PER_WEEK and person_shifts[0]['Equipa'] != 'Volunteer':
                violations += len(person_shifts) - MAX_SHIFTS_PER_WEEK
            elif len(person_shifts) < MIN_SHIFTS_PER_WEEK and person_shifts[0]['Equipa'] != 'Volunteer':
                violations += MAX_SHIFTS_PER_WEEK - len(person_shifts)
            elif len(person_shifts) > MAX_SHIFTS_PER_WEEK_VOLUNTEER and person_shifts[0]['Equipa'] == 'Volunteer':
                violations += len(person_shifts) - MAX_SHIFTS_PER_WEEK_VOLUNTEER
            elif len(person_shifts) < MIN_SHIFTS_PER_WEEK_VOLUNTEER and person_shifts[0]['Equipa'] == 'Volunteer':
                violations += MAX_SHIFTS_PER_WEEK_VOLUNTEER - len(person_shifts)
        return list, violations

    def countJEECMsPerShiftViolations2(self,people_per_shift):
        violations = 0
        list = []
        
        for i, shift in enumerate(people_per_shift):
            if i >= N_SHIFTS_PER_DAY * 2 and i < n_shifts - N_SHIFTS_PER_DAY * 2:
                descontoMin = 6
            else:
                descontoMin = 0
                
            numOfJEECMs = len(shift)
            list.append(numOfJEECMs)
            if (numOfJEECMs > self.shiftMax[i]):
                violations += numOfJEECMs - self.shiftMax[i]
            elif (numOfJEECMs < self.shiftMin[i] - descontoMin):
                violations += self.shiftMin[i] - numOfJEECMs
            
        return list, violations
    
    def countShiftPreferenceViolations2(self, people_per_shift):
        """
        Counts the JEECM-preferences violations in the schedule
        :param JEECMShiftsDict: a dictionary with a separate schedule for each JEECM
        :return: count of violations found
        """
        violations = 0
        for JEECMIndex, shiftPreference in enumerate(self.shiftPreference):
            # duplicate the shift-preference over the days of the period
            # preference = shiftPreference * (self.shiftsPerWeek // n_shifts)
            preference = shiftPreference
            # iterate over the shifts and compare to preferences:
            shifts = people_per_shift[JEECMIndex]
            for pref, shift in zip(preference, shifts):
                if pref == 0 and shift == 1:
                    violations += 1

        return violations
    
    def countlessthan2perDep2(self, people_per_shift):
        violations = 0
                
        counters = {dep: 0 for dep in departments}
        # print(counters)
        
        for i in range(len(people_per_shift)):
            if i >= N_SHIFTS_PER_DAY * 2 and i < n_shifts - N_SHIFTS_PER_DAY * 2: # nao contar com montagem/desm
                people_list = people_per_shift[i]
                for j in range(len(people_list)):
                    people = people_list[j]
                    
                    counters[people['Equipa']] += 1
                
                for dep in departments:
                    if dep != 'Volunteer':
                        if (counters[dep] < NUM_MIN_ELEMENTS_PER_DEP and dep != 'Marketing') or (counters[dep] < 1 and dep == 'Marketing'):
                            violations += 1
                    
                    #penalise a lot of volunteers
                    # if (counters[dep] > MAX_SHIFTS_PER_WEEK_VOLUNTEER and dep == 'Volunteer'):
                    #     violations += 1
                    counters[dep] = 0         

        return violations

    def printScheduleInfo(self, schedule):
        """
        Prints the schedule and violations details
        :param schedule: a list of binary values describing the given schedule
        """
        JEECMShiftsDict = self.getJEECMShifts(schedule)

        print("Schedule for each JEECM:")
        for JEECM in JEECMShiftsDict:  # all shifts of a single JEECM
            print(JEECM, ":", JEECMShiftsDict[JEECM]) 
        
        print("consecutive shift good = ", self.countConsecutiveShiftViolations(JEECMShiftsDict))
        print()

        weeklyShiftsList, violations = self.countShiftsPerWeekViolations(JEECMShiftsDict)
        print("weekly Shifts = ", weeklyShiftsList)
        print("Shifts Per Week Violations = ", violations)
        print()

        totalPerShiftList, violations = self.countJEECMsPerShiftViolations(JEECMShiftsDict)
        print("JEECMs Per Shift = ", totalPerShiftList)
        print("JEECMs Per Shift Violations = ", violations)
        print()

        shiftPreferenceViolations = self.countShiftPreferenceViolations(JEECMShiftsDict)
        print("Shift Preference Violations = ", shiftPreferenceViolations)
        print()
                
        lessthan2perDep = self.countlessthan2perDep(JEECMShiftsDict)
        print("lessthan2perDep violations = ", lessthan2perDep)
        print()

        people_per_shift = [[] for _ in range(n_shifts)]
        # print(n_shifts)
        for member in JEECMShiftsDict:
            shifts = JEECMShiftsDict[member]
            # name = member
            # print(len(shifts))
            
            for i in range(len(self.JEECMs)):
                if self.JEECMs[i] == member:
                    equipa = self.JEECMs_equipas[i]
                        
            for i in range(len(shifts)):
                if shifts[i] == 1:
                    people_per_shift[i].append({'Name': member, 'Equipa': equipa})
                # people_per_shift[i] = sorted(people_per_shift[i], key=lambda x: x['Equipa'][0])
                random.shuffle(people_per_shift[i])
        
        final_distribution = [[] for _ in range(n_shifts)]
        
        vagas = [[] for _ in range(n_shifts)]
        
        for _, row in turnos.iterrows():
            dec = list(eval(row['decorre']))
            for i in range(len(dec)):
                for _ in range(dec[i] * int(row['Num Pessoas'])):
                    vagas[i].append({'turno': row['turnos'], 'pref': row['preferencia']})
        
        # print(vagas)
        
        # distribui sequencialmente
        for i in range(len(people_per_shift)):
            people_list = people_per_shift[i]
            vagas_list = vagas[i]
            
            # Truque para evitar turnos vazios
            # divulg (min4, max6), checkin (min4, max6), cofee (min2, max4)
            if len(vagas_list) == 24:
                vagas_list[10], vagas_list[21] = vagas_list[21], vagas_list[10]
                vagas_list[11], vagas_list[18] = vagas_list[18], vagas_list[11]
                vagas_list[17], vagas_list[22] = vagas_list[22], vagas_list[17]
                vagas_list[19], vagas_list[16] = vagas_list[16], vagas_list[19]
            elif len(vagas_list) == 30:
                plus = 6
                vagas_list[10+plus], vagas_list[21+plus] = vagas_list[21+plus], vagas_list[10+plus]
                vagas_list[11+plus], vagas_list[18+plus] = vagas_list[18+plus], vagas_list[11+plus]
                vagas_list[17+plus], vagas_list[22+plus] = vagas_list[22+plus], vagas_list[17+plus]
                vagas_list[19+plus], vagas_list[16+plus] = vagas_list[16+plus], vagas_list[19+plus]
            
            
            for j in range(len(people_list)):
                people = people_list[j]
                if j < len(vagas_list):
                    final_distribution[i].append({'person': people['Name'], 'turno': vagas_list[j]['turno'], 'Equipa': people['Equipa']}) 
                else:
                    final_distribution[i].append({'person': people['Name'], 'turno': 'Extra', 'Equipa': people['Equipa']}) 
        
        final = []
        kk = 0
        for i in range(N_DIAS_JEEC):
            # print('Dia ', i)
            for j in range(N_SHIFTS_PER_DAY):
                # print('Turno ', j)
                elements = final_distribution[i * N_SHIFTS_PER_DAY + j]
                for element in elements:                    
                    if element['Equipa'] in pref_shift[element['turno']]:
                        pref = True
                    else:
                        pref = False
                        
                    final.append({'Dia': i,'Horario': j, 'Name': element['person'], 'Equipa': element['Equipa'], 'Turno': element['turno'], 'preference': pref, 'id': kk})
                    kk += 1
        
        final = pd.DataFrame(final)
        final.to_excel('distribution_com_extras.xlsx') 
        
        for i in range(N_DIAS_JEEC):
            for j in range(N_SHIFTS_PER_DAY): 
                print(i, j)
                n_trocas = 1
                while n_trocas > 0:
                    troca = []
                    breakk = 0
                                
                    # ver pessoas neste turno sem terem prefencia por este 
                    data = final[(final['Dia'] == i) & (final['Horario'] == j) & (final['preference'] == False)]
                    # print(data)
                    if len(data) > 1:
                        for _, person in data.iterrows():
                            # print('Person: ', person)   
                            for _, k in data.iterrows():
                                # print('k: ', k)
                                if k['Name'] != person['Name'] and person['Equipa'] in pref_shift[k['Turno']]:
                                    troca.append({'a': person['id'], 'b': k['id']})
                                    # print({'a': person['id'], 'b': k['id']})
                                    breakk = 1
                                    break
                                elif k['Name'] != person['Name'] and k['Equipa'] == 'Volunteer' and person['Equipa'] != 'Volunteer' and k['Turno'] == 'CheckIn' and person['Turno'] != 'CheckIn':
                                    troca.append({'a': person['id'], 'b': k['id']})
                                    # Proibir Volunteers no CheckIn
                                    breakk = 1
                                    break
                                    
                            if breakk:
                                break
                    
                    if len(troca) == 0:
                        # Divulgação: prioridade voluntários com uma pessoa de marketing
                        data2 = final[(final['Dia'] == i) & (final['Horario'] == j) & ((final['Equipa'] == 'Marketing') | (final['Equipa'] == 'Volunteer'))]
                        # print(data)
                        n_marketing_in_divulg = len(final[(final['Dia'] == i) & (final['Horario'] == j) & 
                                                        (final['Equipa'] == 'Marketing') & (final['Turno'] == 'Divulgacao')])
                        
                        if len(data2) > 1 and n_marketing_in_divulg < 1:
                            for _, person in data2.iterrows():
                                # print('Person: ', person)   
                                for _, k in data2.iterrows():
                                    # print('k: ', k)
                                    if k['Name'] != person['Name'] and (person['Equipa'] == 'Volunteer' and person['Turno'] == 'Divulgacao' and
                                                                        k['Equipa'] == 'Marketing' and k['Turno'] != 'Divulgacao' and k['Turno'] != 'MarketingTurno'):
                                        troca.append({'a': person['id'], 'b': k['id']})
                                        # print({'a': person['id'], 'b': k['id']})
                                        breakk = 1
                                        break
                                    elif k['Name'] != person['Name'] and (k['Equipa'] == 'Volunteer' and k['Turno'] == 'Divulgacao' and
                                                                        person['Equipa'] == 'Marketing' and person['Turno'] != 'Divulgacao' and person['Turno'] != 'MarketingTurno'):
                                        troca.append({'a': person['id'], 'b': k['id']})
                                        # Proibir Volunteers no CheckIn
                                        breakk = 1
                                        break
                                        
                                if breakk:
                                    break
                    
                    if len(troca) == 0 and len(data) > 1:
                        for _, person in data.iterrows():
                            # print('Person: ', person)   
                            for _, k in data.iterrows():
                                numTurnosperson = len(final[(final['Name'] == person['Name']) & (final['Turno'] != 'Extra')])
                                numTurnosk = len(final[(final['Name'] == k['Name']) & (final['Turno'] != 'Extra')])
                                numTeamMembsInShift = len(data[((data['Equipa'] == person['Equipa']) & (data['Turno'] != 'Extra'))])
                                
                                if k['Name'] != person['Name'] and k['Turno'] == 'Extra' and person['Turno'] != 'Extra' and k['Equipa'] != 'Volunteer' and numTurnosperson > numTurnosk + 1 and (numTeamMembsInShift > 2 or person['Equipa'] == 'Volunteer'):
                                        troca.append({'a': person['id'], 'b': k['id']})
                                        # Proibir Volunteers no CheckIn
                                        breakk = 1
                                        break
                                    
                                elif k['Name'] != person['Name'] and k['Turno'] == 'Extra' and person['Turno'] != 'Extra' and person['Equipa'] == 'Volunteer' and k['Equipa'] != 'Volunteer' and numTurnosperson > MAX_SHIFTS_PER_WEEK_VOLUNTEER:
                                        troca.append({'a': person['id'], 'b': k['id']})
                                        # Proibir Volunteers no CheckIn
                                        breakk = 1
                                        break
                                        
                            if breakk:
                                break
            
                    if len(troca) == 0 and (i > 1 or i < N_SHIFTS_PER_DAY - 2):
                        # Divulgação: prioridade voluntários com uma pessoa de marketing
                        data3 = final[(final['Dia'] == i) & (final['Horario'] == j)]
                        # print(data)
                        
                        if len(data3) > 1:
                            for _, person in data3.iterrows():
                                # print('Person: ', person)   
                                for _, k in data3.iterrows():
                                    # print('k: ', k)
                                    if k['Name'] != person['Name'] and (person['Equipa'] != 'Marketing' and person['Turno'] == 'MarketingTurno' and
                                                                        k['Equipa'] == 'Marketing' and k['Turno'] != 'MarketingTurno'):
                                        troca.append({'a': person['id'], 'b': k['id']})
                                        # print({'a': person['id'], 'b': k['id']})
                                        breakk = 1
                                        break
                                        
                                if breakk:
                                    break
            
                    for tr in troca:     
                        a_index = final.index[final['id'] == tr['a']].tolist()[0] 
                        b_index = final.index[final['id'] == tr['b']].tolist()[0]  
                        
                        # print('Troca', final.at[a_index, 'Turno'], final.at[b_index, 'Turno'])
                        final.at[a_index, 'Turno'], final.at[b_index, 'Turno'] = final.at[b_index, 'Turno'], final.at[a_index, 'Turno']
                        
                        if final.at[a_index, 'Equipa'] in pref_shift[final.at[a_index, 'Turno']]:
                            final.at[a_index, 'preference'] = True
                            # print('Pref true: ', final.at[a_index])
                        else: 
                            final.at[a_index, 'preference'] = False
                            
                        if final.at[b_index, 'Equipa'] in pref_shift[final.at[b_index, 'Turno']]:
                            final.at[b_index, 'preference'] = True
                            # print('Pref true: ', final.at[b_index])
                        else: 
                            final.at[b_index, 'preference'] = False
                
                    n_trocas = len(troca)
                    # print('N trocas = ', n_trocas)
            
        ### Divulgação: prioridade voluntários com uma pessoa de marketing
                
        # final = pd.read_excel('MEGA_TEST/distribution.xlsx')  
    
        # remover extras, com cuidado para o lessthan2
        new_final = pd.DataFrame(columns=final.columns)
        for i in range(N_DIAS_JEEC):
            for j in range(N_SHIFTS_PER_DAY): 
                data = final[(final['Dia'] == i) & (final['Horario'] == j)]
                
                if i == 0 or i == 1 or i == N_DIAS_JEEC - 1 or i == N_DIAS_JEEC - 2:
                    new_final = pd.concat([new_final, data[ (data['Turno'] != 'Extra')]], ignore_index=True)
                
                else:
                    new_final = pd.concat([new_final, data[(data['Equipa'] == 'Volunteer') & (data['Turno'] != 'Extra')]], ignore_index=True)
                    for team in departments:
                        if team != 'Volunteer':
                            numTeamMembsInShift = len(data[((data['Equipa'] == team) & (data['Turno'] != 'Extra'))])
                            new_final = pd.concat([new_final,(data[((data['Equipa'] == team) & (data['Turno'] != 'Extra'))])], ignore_index=True)
                            if numTeamMembsInShift < NUM_MIN_ELEMENTS_PER_DEP:
                                data_to_add = data[(data['Equipa'] == team) & (data['Turno'] == 'Extra')]
                                
                                rg = min((NUM_MIN_ELEMENTS_PER_DEP - numTeamMembsInShift), len(data_to_add))
                                kk = 0
                                for idd, _ in data_to_add.iterrows(): 
                                    if kk < rg:
                                        df_neww = data_to_add[(data_to_add['id'] == idd)]
                                        new_final = pd.concat([new_final,df_neww], ignore_index=True)
                                    
        new_final = new_final[(new_final['Equipa'] == 'Marketing') | ((new_final['Equipa'] != 'Marketing') & (new_final['Turno'] != 'MarketingTurno'))]
        
        new_final.to_excel('distribution.xlsx') 
    
        pessoas_list = list(pessoas['Nome'])
        final_to_computions = [[] for p in pessoas_list]
        
        for i, shift in new_final.iterrows():
            person = shift['Name']
            person_ind = list(pessoas_list).index(person)
            final_to_computions[person_ind].append(shift)
            
        people_per_shift = [[] for _ in range(n_shifts)]
        
        for i, shift in new_final.iterrows():
            people_per_shift[shift['Dia'] * N_SHIFTS_PER_DAY + shift['Horario']].append(shift)
            
        person_shifts = [[0 for i in range(n_shifts)] for p in pessoas_list]
        for i, person in enumerate(final_to_computions):
            for shift in person:
                person_shifts[i][shift['Dia'] * N_SHIFTS_PER_DAY + shift['Horario']] = 1
            
        
        print('Final Distribution:')
        
        print("consecutive shift good = ", self.countConsecutiveShiftViolations2(person_shifts))
        print()

        weeklyShiftsList, violations = self.countShiftsPerWeekViolations2(final_to_computions)
        print("weekly Shifts = ", weeklyShiftsList)
        print("Shifts Per Week Violations = ", violations)
        print()

        totalPerShiftList, violations = self.countJEECMsPerShiftViolations2(people_per_shift)
        print("JEECMs Per Shift = ", totalPerShiftList)
        print("JEECMs Per Shift Violations = ", violations)
        print()

        shiftPreferenceViolations = self.countShiftPreferenceViolations2(person_shifts)
        print("Shift Preference Violations = ", shiftPreferenceViolations)
        print()
                
        lessthan2perDep = self.countlessthan2perDep2(people_per_shift)
        print("lessthan2perDep violations = ", lessthan2perDep)
        print()
    

# problem constants:
HARD_CONSTRAINT_PENALTY = 10000  # the penalty factor for a hard-constraint violation

# Genetic Algorithm constants:
POPULATION_SIZE = 300 # TODO 300
P_CROSSOVER = 0.9  # probability for crossover
P_MUTATION = 0.1   # probability for mutating an individual
MAX_GENERATIONS = 200 # TODO 200
HALL_OF_FAME_SIZE = 30

# set the random seed:
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

toolbox = base.Toolbox()

# create the JEECM scheduling problem instance to be used:
nsp = JEECMSchedulingProblem(HARD_CONSTRAINT_PENALTY)

# define a single objective, maximizing fitness strategy:
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# create the Individual class based on list:
creator.create("Individual", list, fitness=creator.FitnessMin)

# create an operator that randomly returns 0 or 1:
toolbox.register("zeroOrOne", random.randint, 0, 1)

# create the individual operator to fill up an Individual instance:
toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.zeroOrOne, len(nsp))

# create the population operator to generate a list of individuals:
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)


# fitness calculation
def getCost(individual):
    return nsp.getCost(individual),  # return a tuple


toolbox.register("evaluate", getCost)

# genetic operators:
toolbox.register("select", tools.selTournament, tournsize=2)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=1.0/len(nsp))



# Genetic Algorithm flow:
def main():

    # create initial population (generation 0):
    population = toolbox.populationCreator(n=POPULATION_SIZE)

    # prepare the statistics object:
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    # define the hall-of-fame object:
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

    # perform the Genetic Algorithm flow with hof feature added:
    population, logbook = eaSimpleWithElitism(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                                              ngen=MAX_GENERATIONS, stats=stats, halloffame=hof, verbose=True)

    # print best solution found:
    best = hof.items[0]
    print("-- Best Individual = ", best)
    print("-- Best Fitness = ", best.fitness.values[0])
    print()
    print("-- Schedule = ")
    nsp.printScheduleInfo(best)

    # extract statistics:
    minFitnessValues, meanFitnessValues = logbook.select("min", "avg")

    # plot statistics:
    sns.set_style("whitegrid")
    plt.plot(minFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.xlabel('Generation')
    plt.ylabel('Min / Average Fitness')
    plt.title('Min and Average fitness over Generations')
    # plt.show()

if __name__ == "__main__":
    main()
