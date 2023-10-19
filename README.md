Forms: https://docs.google.com/forms/d/1hw4nKhYRcc5oLRINhA9ofr1W6NcNw4JWfUXr0omCbCw/edit#responses

Code for scheduling work shifts during JEEC (Jornadas da Engenharia Eletrotécnica e de Computadores)

It uses a Generative Algorithm

Considered constraints:

hardContstraintViolations = shiftPreferenceViolations + shiftsPerWeekViolations*10 + JEECMsPerShiftViolations

shiftPreferenceViolations: Number of violations in face of forms.xlsx
shiftsPerWeekViolations: Number of violations in face of MAX_SHIFTS_PER_WEEK, MIN_SHIFTS_PER_WEEK, 
MAX_SHIFTS_PER_WEEK_VOLUNTEER and MIN_SHIFTS_PER_WEEK_VOLUNTEER
JEECMsPerShiftViolations: Number of violations in face of the number of member per shift

softContstraintViolations = lessthan2perDep*0.5 - consecutiveShiftViolations*0.01

lessthan2perDep: Number of violations of at least 2 members of each team by shift
consecutiveShiftViolations: Prioritize consecutive shifts


Author: Rui Daniel



Passos de código (para programador):

Questionário -> Horário sem turno atribuído (usa excel das pessoas) -> Distribuir diferentes turnos (usa excel dos turnos)

-Código enfermeiras a funcionar
-Ir buscar funcionarios a um excel
-Ir buscar disponibilidade de funcionarios e aplicar
-Adicionar tipos de turnos diferentes 
-Ir buscar turnos diferentes a um excel
-Mapear pessoas por turnos
-Garantir 2 pessoas de cada departamento sempre
-Minimo de turnos por semana
-Transformar disponibilidades em 0,1.
-Max e min turnos dos voluntários
-Mapear pessoas por turnos com prioridade de departamento pelos turnos (ex: turnos de Marketing têm prioridade de malta do dep de marketing)

-Proibir Volunteers no CheckIn
-Divulgaçao pelo menos 1 de MK, feito com ordenaçao do people por shift (MK aparece sempre primeiro de Vol)


# TODO


-Planear um mega teste

