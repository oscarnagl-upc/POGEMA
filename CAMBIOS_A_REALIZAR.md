Categoría 1 — Problemas del código que debemos corregir

1\. EXP1: la selección del par óptimo ignora la varianza.

El código selecciona por conv\_reward\_mean solamente. El par ganador (eps=0.5, gamma=0.9) tiene la mayor varianza inter-seed del grid (2.008), lo contrario de lo que predecía la hipótesis. El código debería incorporar un criterio combinado o al menos imprimir explícitamente el trade-off recompensa/varianza para que quede documentado en el terminal. Cambio concreto: añadir en la conclusión del EXP1 una línea que muestre el par de mayor recompensa, el par de menor varianza, y si difieren, justificar cuál se elige y por qué.

2\. EXP8: la conclusión no se dibuja en el terminal.

El código dice literalmente "comparar epoch de convergencia entre estrategias. Si el exponencial converge antes en 6x6, confirma la hipótesis." pero nunca evalúa la condición. La condición sí se cumple (IQL 6x6 converge en epoch 61 vs 116) pero el script lo deja al lector. Esto directamente perjudica la puntuación de la ampliación, que necesita "análisis contextualizado" según el enunciado. Cambio concreto: añadir lógica que evalúe y declare explícitamente si H6 se confirma o refuta por algoritmo y tamaño, con el matiz de que "converger antes" no implica "mejor política final".

3\. EXP9: solo 1 seed introduce demasiado ruido.

El caso IQL 4x4 d=0.25 con gap=1.592 (test≈0) podría ser una seed desafortunada. Con 1 seed no hay forma de saber si es un resultado real o ruido. Cambio: usar 3 seeds en EXP9, que sigue siendo barato (son evaluaciones, no entrenamientos completos).

4\. EXP9: la conclusión suaviza los casos malos.

El script concluye que "donde la cobertura baja, el gap crece" pero no señala explícitamente que IQL 4x4 d=0.25 tiene gap=1.592 con test≈0, que es casi no-generalización. Cambio concreto: añadir en el terminal una clasificación explícita: qué configuraciones generalizan bien (gap < 0.3), cuáles regular (0.3-1.0) y cuáles mal (> 1.0).

5\. EXP10: la victoria de Nash+Pareto en d=0.25 no se analiza.

El código detecta que la heterogénea supera a ambas homogéneas pero solo imprime "iguala o supera a la mejor homogénea: no hay penalización por mezclar". Es un análisis insuficiente para una ampliación que necesita contextualización. Cambio: añadir lógica que cuando la heterogénea supera a ambas homogéneas, lo declare como hallazgo sorprendente y lo contextualice mínimamente (la combinación puede ser sinérgica en entornos densos).



Categoría 2 — Problemas de diseño que debemos reconsiderar

6\. EXP4/5: la métrica de inestabilidad intra-run está midiendo JAL-GT como "más inestable" por razones internas, no por no-estacionariedad.

Claude Code lo explica con precisión: JAL-GT oscila porque el bucle learn → update\_policy → nuevo target crea inestabilidad propia, no porque el otro agente cambie su política. Esto significa que nuestra métrica de "firma de no-estacionariedad" no está midiendo lo que pensábamos. No es un bug del código, pero sí afecta cómo debemos interpretar y documentar el resultado. La forma de abordarlo en el código es añadir un comentario/log que distinga entre las dos fuentes de inestabilidad.

7\. EXP1: el heatmap de std inter-seed pierde sentido si el par elegido tiene la mayor varianza.

Si seleccionamos por recompensa máxima e ignoramos la varianza, el heatmap de std es una figura que generamos pero no usamos para tomar decisiones. Debería o bien usarse (criterio combinado) o documentarse explícitamente por qué no se usa. Cambio de diseño: añadir en experiment\_1\_hyperparams la impresión del par de mínima varianza junto al de máxima recompensa.

8\. EXP7 (contraste situacional Pareto vs Minimax en 4x4 y 10x10): su propósito era ver si el ranking depende del tamaño, pero los datos de EXP4/5 ya cubren JAL-GT en los tres tamaños. Con los resultados de EXP6 y EXP4/5 tienes más que suficiente para argumentar sobre situaciones adecuadas para cada concepto. EXP7 añade dos runs de JAL-GT en 4x4 y 10x10 con solo Pareto y Minimax, y la conclusión fue "la diferencia NO se amplifica con el tamaño", que es un resultado poco llamativo. Por lo tanto debemos eliminar EXP7 (y corregir por lo tanto los números de los experimentos).

9\. Número de seeds: debemos reducir el número de seeds usadas en los experimentos para agilizar el tiempo de ejecución según la siguiente tabla:

|Bloque (números de experimento antiguos - antes de eliminar EXP7)|Seeds|
|-|-|
|EXP1 (grid ε/γ)|5|
|EXP2 (transferencia γ)|3|
|EXP3 (validación 10x10)|3|
|EXP4/5 (IQL vs JAL-GT)|3|
|EXP6 (conceptos)|3|
|EXP8 (decay)|5|
|EXP9 (generalización)|3|
|EXP10 (heterogéneas)|3|

10\. Flag --extra: los experimentos 2 y 3, al ocuparse únicamente de validar la transferencia de hiperparámetros, solo se ejecutarán si el flag --complete (nuevo flag) esta activo. Por lo tanto si se ejecuta experiments.py sin flags no se deben realizar los experimentos 2 y 3, asumiendo la correcta transferencia de hiperparámetros.

