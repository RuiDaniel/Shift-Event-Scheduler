from deap import tools
from deap import algorithms
from deap import base
from deap import creator

import random

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd

N_WEEKS_JEEC = 1
N_DIAS_JEEC = 2
MAX_SHIFTS_PER_WEEK = 20
MIN_SHIFTS_PER_WEEK = 5
N_SHIFTS_PER_DAY = 10
departments = ['WebDev', 'Speakers', 'Business', 'Coordinator', 'Logistics', 'Marketing']
n_shifts = N_DIAS_JEEC * N_SHIFTS_PER_DAY
NUM_MIN_ELEMENTS_PER_DEP = 2

# Generate pessoas.xlsx based on forms.xlsx
def use_forms():
    name = 'forms'
    read_file = pd.read_excel (name + ".xlsx")
    read_file.to_csv (name + ".csv", 
                    index = None,
                    header=True, encoding='utf-8')

    forms = pd.read_csv(name + ".csv", encoding='utf-8') 

    dias = ['Dia 1', 'Dia 2', 'Dia 3', 'Dia 4', 'Dia 5', 'Dia 6', 'Dia 7', 'Dia 8', 'Dia 9']

    turnoss= ['_8h_9h', '_9h_10h', '_10h_11h', '_11h_12h', '_12h_13h', '_13h_14h', '_14h_15h', '_15h_16h', '_16h_17h', '_17h_18h']

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

name = 'pessoas'
read_file = pd.read_excel (name + ".xlsx")
read_file.to_csv (name + ".csv", 
                  index = None,
                  header=True, encoding='utf-8')

name = 'turnos'
read_file = pd.read_excel (name + ".xlsx")
read_file.to_csv (name + ".csv", 
                  index = None,
                  header=True, encoding='utf-8')

pessoas = pd.read_csv('pessoas.csv', encoding='utf-8') 
turnos = pd.read_csv('turnos.csv', encoding='utf-8') 


        
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

class NurseSchedulingProblem:
    """This class encapsulates the Nurse Scheduling problem
    """

    def __init__(self, hardConstraintPenalty):
        """
        :param hardConstraintPenalty: the penalty factor for a hard-constraint violation
        """
        self.hardConstraintPenalty = hardConstraintPenalty

        # list of nurses:
        self.nurses = pessoas['Nome']
        self.nurses_equipas = pessoas['Equipa']

        # nurses' respective shift preferences - morning, evening, night:
        self.shiftPreference = pessoas['Disponibilidade']

        # min and max number of nurses allowed for each shift - morning, evening, night:
        turnos_head_dec = str(turnos.head(1)['decorre']).split()[1]
        
        self.shiftMin = [0 for _ in turnos_head_dec.split(sep=',')]
        self.shiftMax = [0 for _ in turnos_head_dec.split(sep=',')]

        for _, row in turnos.iterrows():
            dec = list(eval(row['decorre']))
            
            for i in range(len(dec)):
                self.shiftMin[i] += dec[i] * int(row['Num Pessoas'])
                self.shiftMax[i] += dec[i] * int(row['Num Pessoas'])

        # max shifts per week allowed for each nurse
        self.maxShiftsPerWeek = MAX_SHIFTS_PER_WEEK
        self.minShiftsPerWeek = MIN_SHIFTS_PER_WEEK

        # number of weeks we create a schedule for:
        self.weeks = N_WEEKS_JEEC

        # useful values:
        self.shiftPerDay = len(self.shiftMin)
        self.shiftsPerWeek = N_DIAS_JEEC * self.shiftPerDay

        
    def __len__(self):
        """
        :return: the number of shifts in the schedule
        """
        return len(self.nurses) * self.shiftsPerWeek * self.weeks


    def getCost(self, schedule):
        """
        Calculates the total cost of the various violations in the given schedule
        ...
        :param schedule: a list of binary values describing the given schedule
        :return: the calculated cost
        """

        if len(schedule) != self.__len__():
            raise ValueError("size of schedule list should be equal to ", self.__len__())

        # convert entire schedule into a dictionary with a separate schedule for each nurse:
        nurseShiftsDict = self.getNurseShifts(schedule)

        # count the various violations:
        # consecutiveShiftViolations = self.countConsecutiveShiftViolations(nurseShiftsDict)
        shiftsPerWeekViolations = self.countShiftsPerWeekViolations(nurseShiftsDict)[1]
        nursesPerShiftViolations = self.countNursesPerShiftViolations(nurseShiftsDict)[1]
        shiftPreferenceViolations = self.countShiftPreferenceViolations(nurseShiftsDict)
        lessthan2perDep = self.countlessthan2perDep(nurseShiftsDict)

        # calculate the cost of the violations:
        hardContstraintViolations = nursesPerShiftViolations + shiftsPerWeekViolations + lessthan2perDep
        softContstraintViolations = shiftPreferenceViolations

        return self.hardConstraintPenalty * hardContstraintViolations + softContstraintViolations

    def getNurseShifts(self, schedule):
        """
        Converts the entire schedule into a dictionary with a separate schedule for each nurse
        :param schedule: a list of binary values describing the given schedule
        :return: a dictionary with each nurse as a key and the corresponding shifts as the value
        """
        shiftsPerNurse = self.__len__() // len(self.nurses)
        nurseShiftsDict = {}
        shiftIndex = 0

        for nurse in self.nurses:
            nurseShiftsDict[nurse] = schedule[shiftIndex:shiftIndex + shiftsPerNurse]
            shiftIndex += shiftsPerNurse

        return nurseShiftsDict

    def countConsecutiveShiftViolations(self, nurseShiftsDict):
        """
        Counts the consecutive shift violations in the schedule
        :param nurseShiftsDict: a dictionary with a separate schedule for each nurse
        :return: count of violations found
        """
        violations = 0
        # iterate over the shifts of each nurse:
        for nurseShifts in nurseShiftsDict.values():
            # look for two cosecutive '1's:
            for shift1, shift2 in zip(nurseShifts, nurseShifts[1:]):
                if shift1 == 1 and shift2 == 1:
                    violations += 1
        return violations

    def countShiftsPerWeekViolations(self, nurseShiftsDict):
        """
        Counts the max-shifts-per-week violations in the schedule
        :param nurseShiftsDict: a dictionary with a separate schedule for each nurse
        :return: count of violations found
        """
        violations = 0
        weeklyShiftsList = []
        # iterate over the shifts of each nurse:
        for nurseShifts in nurseShiftsDict.values():  # all shifts of a single nurse
            # iterate over the shifts of each weeks:
            for i in range(0, self.weeks * self.shiftsPerWeek, self.shiftsPerWeek):
                # count all the '1's over the week:
                weeklyShifts = sum(nurseShifts[i:i + self.shiftsPerWeek])
                weeklyShiftsList.append(weeklyShifts)
                if weeklyShifts > self.maxShiftsPerWeek:
                    violations += weeklyShifts - self.maxShiftsPerWeek
                elif weeklyShifts < self.minShiftsPerWeek:
                    violations += self.minShiftsPerWeek - weeklyShifts

        return weeklyShiftsList, violations

    def countNursesPerShiftViolations(self, nurseShiftsDict):
        """
        Counts the number-of-nurses-per-shift violations in the schedule
        :param nurseShiftsDict: a dictionary with a separate schedule for each nurse
        :return: count of violations found
        """
        # sum the shifts over all nurses:
        totalPerShiftList = [sum(shift) for shift in zip(*nurseShiftsDict.values())]

        violations = 0
        # iterate over all shifts and count violations:
        for shiftIndex, numOfNurses in enumerate(totalPerShiftList):
            dailyShiftIndex = shiftIndex % self.shiftPerDay  # -> 0, 1, or 2 for the 3 shifts per day
            if (numOfNurses > self.shiftMax[dailyShiftIndex]):
                violations += numOfNurses - self.shiftMax[dailyShiftIndex]
            elif (numOfNurses < self.shiftMin[dailyShiftIndex]):
                violations += self.shiftMin[dailyShiftIndex] - numOfNurses

        return totalPerShiftList, violations

    def countShiftPreferenceViolations(self, nurseShiftsDict):
        """
        Counts the nurse-preferences violations in the schedule
        :param nurseShiftsDict: a dictionary with a separate schedule for each nurse
        :return: count of violations found
        """
        violations = 0
        for nurseIndex, shiftPreference in enumerate(self.shiftPreference):
            # duplicate the shift-preference over the days of the period
            # preference = shiftPreference * (self.shiftsPerWeek // self.shiftPerDay)
            preference = shiftPreference
            # iterate over the shifts and compare to preferences:
            shifts = nurseShiftsDict[self.nurses[nurseIndex]]
            for pref, shift in zip(preference, shifts):
                if pref == 0 and shift == 1:
                    violations += 1

        return violations
    
    def countlessthan2perDep(self, nurseShiftsDict):
        violations = 0
        
        people_per_shift = [[] for _ in range(n_shifts)]
        
        counters = {dep: 0 for dep in departments}
        # print(counters)
        
        for member in nurseShiftsDict:
            shifts = nurseShiftsDict[member]
            # name = member
            # print(len(shifts))
            for i in range(len(shifts)):
                if shifts[i] == 1:
                    people_per_shift[i].append(member)
        
        for i in range(len(people_per_shift)):
            people_list = people_per_shift[i]
            for j in range(len(people_list)):
                people = people_list[j]
                # print(people)
                # Find index of person and return team
                # Funciona se nao houver nomes repetidos
                for i in range(len(self.nurses)):
                    if self.nurses[i] == people:
                        equipa = self.nurses_equipas[i]
                
                counters[equipa] += 1
            
            for dep in departments:
                if counters[dep] < NUM_MIN_ELEMENTS_PER_DEP:
                    violations += 1
                counters[dep] = 0
                

        return violations

    def printScheduleInfo(self, schedule):
        """
        Prints the schedule and violations details
        :param schedule: a list of binary values describing the given schedule
        """
        nurseShiftsDict = self.getNurseShifts(schedule)

        print("Schedule for each nurse:")
        for nurse in nurseShiftsDict:  # all shifts of a single nurse
            print(nurse, ":", nurseShiftsDict[nurse]) 

        # print("consecutive shift violations = ", self.countConsecutiveShiftViolations(nurseShiftsDict))
        # print()

        weeklyShiftsList, violations = self.countShiftsPerWeekViolations(nurseShiftsDict)
        print("weekly Shifts = ", weeklyShiftsList)
        print("Shifts Per Week Violations = ", violations)
        print()

        totalPerShiftList, violations = self.countNursesPerShiftViolations(nurseShiftsDict)
        print("Nurses Per Shift = ", totalPerShiftList)
        print("Nurses Per Shift Violations = ", violations)
        print()

        shiftPreferenceViolations = self.countShiftPreferenceViolations(nurseShiftsDict)
        print("Shift Preference Violations = ", shiftPreferenceViolations)
        print()
        
        lessthan2perDep = self.countlessthan2perDep(nurseShiftsDict)
        
        lessthan2perDep = self.countlessthan2perDep(nurseShiftsDict)
        print("lessthan2perDep violations = ", lessthan2perDep)
        print()

        people_per_shift = [[] for _ in range(n_shifts)]
        # print(n_shifts)
        for member in nurseShiftsDict:
            shifts = nurseShiftsDict[member]
            # name = member
            # print(len(shifts))
            
            for i in range(len(self.nurses)):
                if self.nurses[i] == member:
                    equipa = self.nurses_equipas[i]
                        
            for i in range(len(shifts)):
                if shifts[i] == 1:
                    people_per_shift[i].append({'Name': member, 'Equipa': equipa})
        
        # print(people_per_shift)
        
        final_distribution = [[] for _ in range(n_shifts)]
        
        vagas = [[] for _ in range(n_shifts)]
        
        for _, row in turnos.iterrows():
            dec = list(eval(row['decorre'])) * N_DIAS_JEEC
            for i in range(len(dec)):
                for _ in range(dec[i] * int(row['Num Pessoas'])):
                    vagas[i].append({'turno': row['turnos'], 'pref': row['preferencia']})
        
        # print(vagas)
        
        for i in range(len(people_per_shift)):
            people_list = people_per_shift[i]
            vagas_list = vagas[i]
            
            for j in range(min(len(people_list), len(vagas_list))):
                people = people_list[j]
                final_distribution[i].append({'person': people['Name'], 'turno': vagas_list[j]['turno'], 'Equipa': people['Equipa']}) 
                
        
        final = []
        for i in range(N_DIAS_JEEC):
            # print('Dia ', i)
            for j in range(N_SHIFTS_PER_DAY):
                # print('Turno ', j)
                elements = final_distribution[i * N_SHIFTS_PER_DAY + j]
                for element in elements:
                    print(element['person'], element['turno'])
                    final.append({'Dia': i,'Horario': j, 'Name': element['person'], 'Equipa': element['Equipa'], 'Turno': element['turno']})
                    
        df = pd.DataFrame(final)
        df.to_excel('distribution.xlsx', index=False)  
                
                
                
        
        


# problem constants:
HARD_CONSTRAINT_PENALTY = 10000  # the penalty factor for a hard-constraint violation

# Genetic Algorithm constants:
POPULATION_SIZE = 300
P_CROSSOVER = 0.9  # probability for crossover
P_MUTATION = 0.1   # probability for mutating an individual
MAX_GENERATIONS = 200
HALL_OF_FAME_SIZE = 30

# set the random seed:
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

toolbox = base.Toolbox()

# create the nurse scheduling problem instance to be used:
nsp = NurseSchedulingProblem(HARD_CONSTRAINT_PENALTY)

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
    plt.show()

        
    









if __name__ == "__main__":
    main()