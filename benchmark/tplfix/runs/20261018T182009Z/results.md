# TPL-FIX OLD vs NEW

NOT A MEASUREMENT (agent resident)

| Voice command | OLD parsed | NEW parsed | Same actions | OLD leaks | NEW leaks |
|---|---|---|---|---|---|
| Find Chiara and tell her the pizza is here | True | True | True | 0 | 0 |
| Go to the bathroom | True | True | True | 0 | 0 |
| Patrol the whole apartment | True | True | True | 0 | 0 |
| Find my phone in the living room | True | True | True | 0 | 0 |
| Come back | True | True | True | 0 | 0 |
| Tell Abdel the meeting starts in ten minutes | True | True | True | 0 | 0 |
| Go to the kitchen and say lunch is ready | True | True | True | 0 | 0 |
| Check the bedroom and the hallway | True | True | True | 0 | 0 |
| Where are my glasses | True | True | True | 0 | 0 |
| Say good night | True | True | True | 0 | 0 |
| Find someone and tell them the door is open | True | True | True | 0 | 0 |
| Go to the living room then come back | True | True | True | 0 | 0 |
| Look for the cat in the bedroom | True | True | True | 0 | 0 |
| Tell Chiara I love her | True | True | True | 0 | 0 |
| Blue banana seven | True | True | False | 0 | 0 |

| Chat prompt | OLD leaks | NEW leaks | OLD repeated | NEW repeated |
|---|---|---|---|---|
| Hello! Introduce yourself in one sentence. | 0 | 1 | False | False |
| What is two plus three? Answer briefly. | 0 | 2 | False | False |
| Name three common household objects. | 0 | 1 | False | False |
| Explain why a robot should stop near an obstacle in one sentence. | 0 | 1 | False | False |
| Write a friendly good morning greeting. | 0 | 1 | False | False |

Repeated = an exact eight-word span repeated without overlap; read raw replies for quality.
