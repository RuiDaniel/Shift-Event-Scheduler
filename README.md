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

-Mega teste

PROBLEMAS:

-Voluntarios a ficarem com pref verdadeira em turnos extra: corrigido
-Demasiadas pessoas em turnos extra (principalmente Voluntarios)
-So é preciso 1 pessoa de mk em cada turno

Regras:
No mínimo 2 pessoas de business durante o dia (sem contar com o Diogo)
Voluntários não fazer turnos de checkin
No mínimo 2 pessoas de marketing (a contar com a Inês)
No mínimo 2 pessoas de webdev (sem contar com o José)
No mínimo 2 pessoas de logistics (para além do Rodrigo)
Checkin: prioridade business ou speakers ou logística
Auditório: prioridade logística
Almoços: prioridade business e logística
Divulgação: prioridade voluntários com uma pessoa de marketing


2º mega teste

-Demasiadas pessoas em turnos extra (principalmente Voluntarios): correçao feito com 10*shiftsPerWeekViolations


9 dias de turnos
Voluntários: 20-30
Turnos: 1h30
Começo dos turnos: 9h, Termino dos turnos: 19h30 (havendo segurança)
Máximo de turnos por pessoa: Não há
Mínimo de turnos por pessoa: 3 para voluntários, 5 para equipa
Turnos: 6 Checkin, Montagem/Desmontagem (mínimo 6), Coffee Break (mínimo 2, max 4), Divulgação (4 a 6 pessoas), Refeições (4 a 6 pessoas), Auditório (2 pessoas), Workshops (4 pessoas)


-Novo Turno Marketing(2), lessthan2perDep também para marketing
-Pessoas de MK preferencia por MKTurno em vez de divulgaçao
-Apagar consecutive de um dia para o outro


# TODO

-Por no git da jeec
-Correr MegaTeste 23-10



