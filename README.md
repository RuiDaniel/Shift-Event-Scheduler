Code for scheduling work shifts in an event:

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
