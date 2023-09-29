Forms: https://docs.google.com/forms/d/1hw4nKhYRcc5oLRINhA9ofr1W6NcNw4JWfUXr0omCbCw/edit#responses

Requisitos:!!!!!!!!!!!!!
No excel do forms:
-Nomes colunas: Dia 1, Dia 2, Dia 3, ...
-Converter 8h-9h para _8h_9h, 9h-10h para _9h_10h, ...

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

# TODO:

-Mapear pessoas por turnos com prioridade de departamento pelos turnos (ex: turnos de Marketing têm prioridade de malta do dep de marketing)

estratégia: agrupar por departamento em cada turno

exemplo:

Turno 1: 2 vagas
Turno 2: 2 vagas

pessoas: WD,WD,MK,SP

WD,WD
MK
SP

WD prioridade Turno 2


for turno in Turnos:
    while group prioritario not empty and turno not full:
        turno.append(elemento grupo prioritario)

o resto distribuir sequencialmente

Turno 2: WD,WD
Turno 1: MK,SP