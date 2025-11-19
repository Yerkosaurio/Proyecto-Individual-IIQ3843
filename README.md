# Proyecto-Individual-IIQ3843
Comportamiento T° 1D
1. Contexto y Relevancia del Proyecto

Contexto

El almacenamiento de energía térmica es una tecnología clave para la eficiencia energética, permitiendo acumular calor en un fluido para su uso posterior. Este proyecto se enfoca en estanques de almacenamiento térmico en un contexto industrial o residencial de alta demanda, como sistemas de calefacción distrital.

La simulación se sitúa en un escenario geográfico demandante, como puede ser la zona austral de Chile, donde las temperaturas ambientales nocturnas pueden ser de -20°C, mientras que la radiación solar diurna puede calentar las superficies expuestas a más de 40°C.

Relevancia y brecha a resolver:
El desafío principal en la simulación de sistemas como estos es el equilibrio entre precisión y costo computacional.
Modelos 3D (CFD), son de alta fidelidad, pero su costo computacional es alto para simulaciones de larga duración (días o semanas).

Modelos 1D Simples, son rápidos, pero a menudo fallan al simplificar excesivamente la física, sufriendo de difusión numérica lo cual inventa mezclas y destruye la estratificación térmica.

La brecha que este proyecto resuelve es desarrollar un modelo 1D que sea computacionalmente rápido, pero que represente adecuadamente la física del fenómeno. Esto se logra ajustando el modelo para incluir los efectos de la convección natural dentro de una ecuación de conducción simple, permitiendo analizar con precisión la evolución temporal de la temperatura bajo condiciones de borde transientes y realistas.


2. Descripción del Sistema

El sistema modelado es un estanque cilíndrico vertical de almacenamiento térmico, con una altura L = 7,5 m y un diámetro 5 = 5 m.

El modelo es unidimensional 1D, analizando la variación de temperatura T(x,t) únicamente a lo largo del eje vertical x, donde x=0 es la base y x=L es la superficie superior.

Las condiciones físicas del sistema son:
Fluido: El estanque contiene aceite térmico (Mobiltherm 605). Se elige este fluido sobre el agua para operar de forma segura en el contexto austral, ya que previene el congelamiento (punto de fusión en -27°C) y elimina el riesgo de ebullición y presurización (punto de ebullición mayor a 300°C).

Condición Inicial (T0): El estanque se encuentra uniformemente a 80°C, una temperatura estándar para calefacción.

Borde Base (x=0): Se asume que la base está perfectamente aislada (adiabática).

Borde Superior (x=L): La superficie está expuesta a la temperatura ambiente variable, que sigue un ciclo diario realista (mínimo de -20°C a las 3 AM, máximo de 45°C a las 3 PM).


3. Modelo Matemático

El comportamiento térmico del estanque se describe mediante la ecuación de conducción de calor unidimensional en régimen transitorio. Este modelo asume que la variación de temperatura en las direcciones radial y angular es despreciable comparada con la variación vertical, debido a la estratificación y geometría del estanque.


3.1 Ecuación Gobernante

La evolución temporal de la temperatura T(x,t) está regida por:
dT/dt = Alpha eff * d^2T/ d^2x

Donde:

T: Temperatura del fluido en °C.

t: Tiempo en segundos.

x: Posición vertical (m), donde x=0 es la base y x=L es la superficie.

Alpha eff: Difusividad térmica efectiva del medio (m^2/s).


3.2 Parámetros Termofísicos 

Se utiliza Mobiltherm 605 como fluido de trabajo, almacenado a 80°C. Las propiedades base consideradas son:

Densidad (rho): 835 kg/m^3 

Capacidad Calorífica (Cp): 2000 J/(kg*K)

Conductividad Térmica Base (k base): 0.13 W/(m*K)

Parámetros del Modelo (Ajuste por Convección)

Para compensar las limitaciones de un modelo 1D de conducción pura y representar los efectos de mezcla por convección natural dentro del estanque, se introduce una conductividad térmica efectiva (k eff) aumentada experimentalmente por un factor de 200.

Conductividad Efectiva:

K eff = k base * 200 = 26.0 W/(m*K)

Difusividad Térmica Efectiva Resultante:

Alpha eff = k eff/(rho*Cp) = 1,557*10^5 m^2/s


3.3 Condiciones iniciales y de contorno

El sistema diferencial se cierra con las siguientes condiciones que definen el estado inicial y la interacción con los bordes:

a) Condición Inicial (Estado del Estanque en t=0):

El fluido comienza a una temperatura uniforme de almacenamiento.

T(x, 0) = 80 °C para todo x perteneciente a [0, L]

b) Condición de Borde Inferior (x=0, Base):

Se asume que la base del estanque está perfectamente aislada (adiabática). Matemáticamente, esto corresponde a una condición de Neumann homogénea (flujo nulo).

dT/dt evaluado en x =0 es 0

c) Condición de Borde Superior (x=L, Superficie):

La superficie está expuesta a una temperatura ambiente variable que sigue un ciclo diario realista, con una mínima de -20°C a las 03:00 AM y una máxima de 45°C a las 15:00 PM. Esto se modela como una condición de Dirichlet dependiente del tiempo:

T(L,t) = T prom – A*cos(2Pi(t-t desfase) / tau)

Donde los coeficientes de la oscilación térmica son:

Temperatura Promedio (T prom): 12,5°C

Amplitud (A): 32,5 °C

Periodo (tau): 86400 s (24 horas)

Desfase (t desfase): 10800 s (3 horas, para fijar el mínimo a las 3 AM)


4. Métodos Numéricos: Discretización y Solución

Para resolver la ecuación diferencial parcial (EDP) gobernante, se emplea el Método de Líneas (MOL). Esta estrategia numérica consiste en discretizar el dominio espacial para transformar la EDP en un sistema de Ecuaciones Diferenciales Ordinarias, manteniendo la variable temporal continua.


4.1. Discretización del Dominio Espacial

El dominio vertical del estanque, de altura L = 7,5 m, se divide en una malla uniforme de N = 50 nodos.
Espaciamiento de la malla (Delta x):

Delta x = L / (N-1)

Posición de los nodos:

La posición de un nodo i está dada por x i = i * Delta x, donde i = 0, 1, …, N-1 

i=0: Base del estanque.

i=N-1: Superficie del estanque.

4.2 Aproximación por Diferencias Finitas

Para aproximar la segunda derivada espacial d^2T/ d^2x en cada nodo interno i, se utiliza el esquema de diferencias finitas centradas de segundo orden.

La aproximación algebraica es:

d^2T/ d^2x = [T(i+1) – 2*T(i) + T(i-1)]/Delta x^2

Sustituyendo esta aproximación en la ecuación gobernante, obtenemos la EDO para la variación temporal de la temperatura en cada nodo interno (i entre 1 y N-2):

dTi/dt = Alpha eff / Delta x^2 * [T(i+1) – 2*T(i) + T(i-1)]


4.3 Tratamiento de las Condiciones de Borde

Borde Inferior (Base, i=0): Condición de Neumann

La base es adiabática (dT/dt= 0). Para implementar esto numéricamente sin perder precisión de segundo orden, se utiliza el método del Nodo Fantasma.

Imaginamos un nodo ficticio i=-1 fuera del dominio. Aplicando diferencias centradas a la primera derivada en x=0:

[T(1)-T(-1)] / (2*Delta x) = 0 lo que implica T(-1) = T(1)

Sustituyendo T(-1) en la ecuación general para el nodo 0, obtenemos la EDO modificada:

dT0/dt = 2* Alpha eff / Delta x^2 * [T(1) – T(0)]

 Borde Superior (Superficie, i=N-1): Condición de Dirichlet
 
La temperatura en el último nodo T(N-1) no es una incógnita diferencial, sino un valor forzado dependiente del tiempo T superficie (t).

Este valor actúa como condición de frontera para el penúltimo nodo (i=N-2). La ecuación para este nodo es:

dT(N-2)/dt = Alpha eff / Delta x^2 *[T superficie (t) – 2T(N-2) + T(N-3)]


4.4 Integración Temporal

El resultado de la discretización espacial es un sistema de N-1 Ecuaciones Diferenciales Ordinarias acopladas de la forma dT/dt = f(t, T)

Para resolver este sistema, se utiliza el integrador RK45 de la librería scipy.integrate.
Método: Runge-Kutta-Fehlberg explícito de orden 5(4).

Característica clave: Posee control de paso de tiempo adaptativo. El algoritmo ajusta automáticamente el tamaño del paso temporal (Delta t) según la tasa de cambio de la temperatura, reduciendo el paso cuando los gradientes térmicos son elevados para garantizar la estabilidad y precisión numérica.


5. Explicación Detallada del Código

El script de Python implementa el modelo numérico descrito anteriormente.

import numpy as np

from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt

import matplotlib.style as style


 Configuración estética de los gráficos:
 
style.use('seaborn-v0_8-whitegrid')


5.1 Definición del Sistema de EDOs (Método de Líneas)

def model(t, T, k, rho, cp, L, N):

Esta función define el sistema de EDOs (dT/dt) para el solver.

't' es el tiempo actual, 'T' es el vector de temperaturas (T_0 a T_N-2).

El resto son parámetros físicos pasados por 'args'.    


Parámetros Numéricos 

alpha = k / (rho * cp)  donde k es k eff

dx = L / (N - 1)

     
C.B. de Superficie (Dirichlet Transiente):

Calcula la T° de la superficie (nodo N-1) en el tiempo 't'

T_mean = 12.5

A = 32.5

tau = 24 * 3600

shift_seconds = 3 * 3600 

T_superficie = T_mean - A * np.cos(2 * np.pi * (t - shift_seconds) / tau)

     
Cálculo de Derivadas (dT/dt) :

Crea un vector vacío para guardar las derivadas de los N-1 nodos:

dTdt = np.zeros(N - 1)

     
Ecuación para el Nodo 0 (Base Adiabática, C.B. Neumann):

dTdt[0] = 2.0 * alpha * (T[1] - T[0]) / (dx**2)

     
Ecuaciones para Nodos Internos (i=1 hasta N-3):

for i in range(1, N - 2):
     
  dTdt[i] = alpha * (T[i+1] - 2*T[i] + T[i-1]) / (dx**2)
      
     
Ecuación para el Nodo N-2 (Último nodo a resolver):

Conecta el sistema al valor de T_superficie

dTdt[N-2] = alpha * (T_superficie - 2*T[N-2] + T[N-3]) / (dx**2)

     
Devuelve el vector de derivadas al solver

  return dTdt

5.2 Configuración de la Simulación

Parámetros Físicos (Mobiltherm 605 a 80°C):

rho = 835

cp = 2000

L = 7.5      Altura (m)

N = 50      Número de nodos

k_base = 0.13  Conductividad real


Parámetros Numéricos:

x_plot = np.linspace(0, L, N)  Eje X para graficar


Condición Inicial (C.I.):

T_inicial_uniforme = 80.0

Vector de estado inicial (N-1 elementos, de T_0 a T_N-2):

T_vec_inicial = np.full(N - 1, T_inicial_uniforme)


Configuración del Tiempo:

t_inicio = 0

t_fin = 1 * 24 * 3600  Simula 1 día

Puntos de tiempo donde se guardará la solución:

t_puntos = np.linspace(t_inicio, t_fin, 100) 


Parámetro del Modelo (k-efectivo):

k_efectivo = k_base * 200   k_eff = 26.0


5.3 Ejecución del Solver

sol_efectivo = solve_ivp(

  fun=model,                     La función que define las EDOs
  
  t_span=(t_inicio, t_fin),      Rango de tiempo de la simulación
  
  y0=T_vec_inicial,              El vector de condición inicial (T a t=0)
  
  t_eval=t_puntos,               Tiempos específicos para guardar la salida
  
  method='RK45',                 Integrador Runge-Kutta 4(5) adaptativo
  
  args=(k_efectivo, rho, cp, L, N)  Parámetros físicos para pasar a 'model'
  
)

'sol_efectivo' es un objeto que contiene los resultados:

sol_efectivo.t -> Tiempos (vector)

sol_efectivo.y -> Temperaturas (matriz de [N-1, 100])


5.4 Post-Procesamiento y Visualización

Crear la figura con 2 subplots (1 fila, 2 columnas)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10), sharey=True) 


Parámetros para recalcular T_superficie (solo para graficar)

T_mean = 12.5

A = 32.5

tau = 24 * 3600

shift_seconds = 3 * 3600


Tramo 1: Gráfico de Aumento (3 AM - 3 PM) 

tiempos_aumento_h = np.arange(3, 15 + 3, 3)  Horas a plotear: [3, 6, 9, 12, 15]

cmap_aumento = plt.cm.plasma

norm_aumento = plt.Normalize(vmin=min(tiempos_aumento_h), vmax=max(tiempos_aumento_h))


for t_h in tiempos_aumento_h:

     Encuentra el índice (0-99) más cercano a la hora deseada:
     t_idx = np.argmin(np.abs(sol_efectivo.t - t_h * 3600))
     Extrae el perfil de temperatura (N-1 puntos) de la solución
     T_perfil_parcial = sol_efectivo.y[:, t_idx]
     Recalcula T_superficie en ese instante exacto
     t_s = sol_efectivo.t[t_idx]
     T_superficie = T_mean - A * np.cos(2 * np.pi * (t_s - shift_seconds) / tau)
     Une el perfil simulado (N-1) con el de borde (1) para un gráfico completo (N)
     T_perfil_completo = np.append(T_perfil_parcial, T_superficie)
     Dibuja la línea en el subplot 1
     ax1.plot(x_plot, T_perfil_completo, color=cmap_aumento(norm_aumento(t_h)), linewidth=2.5) 

ax1.set_title('Tramo 1: Aumento T° Ambiente (3 AM - 3 PM)', fontsize=16)

ax1.set_xlabel('Altura estanque (m)', fontsize=14)

ax1.set_ylabel('Temperatura ($^{\circ}$C)', fontsize=14)

ax1.grid(True, linestyle='--')


Tramo 2: Gráfico de Descenso (3 PM - 3 AM)

Mapeo de horas: {Hora real: Valor para el color}

Se usa '27' para 3 AM para que el color sea el final del ciclo

tiempos_descenso_h_map = {15: 15, 18: 18, 21: 21, 24: 24, 3: 27} 

cmap_descenso = plt.cm.viridis

norm_descenso = plt.Normalize(vmin=15, vmax=27) 


for t_h_real, t_h_color in tiempos_descenso_h_map.items():

 La lógica es idéntica al Tramo 1, pero usa el mapeo de horas
 
     t_idx = np.argmin(np.abs(sol_efectivo.t - t_h_real * 3600))
     T_perfil_parcial = sol_efectivo.y[:, t_idx]
     t_s = sol_efectivo.t[t_idx]
     T_superficie = T_mean - A * np.cos(2 * np.pi * (t_s - shift_seconds) / tau)
     T_perfil_completo = np.append(T_perfil_parcial, T_superficie)
     
 Dibuja la línea en el subplot 2
 
     ax2.plot(x_plot, T_perfil_completo, color=cmap_descenso(norm_descenso(t_h_color)), linewidth=2.5)

ax2.set_title('Tramo 2: Descenso T° Ambiente (3 PM - 3 AM)', fontsize=16)

ax2.set_xlabel('Altura estanque (m)', fontsize=14)

ax2.grid(True, linestyle='--')


cbar2.set_ticks([15, 18, 21, 24, 27]) 

cbar2.set_ticklabels(['3 PM', '6 PM', '9 PM', '12 AM', '3 AM']) 


6. Resultados y Conclusiones

Los gráficos generados muestran la evolución del perfil de temperatura dentro del estanque.
Se observa que la temperatura base (x=0) permanece inalterada a 80°C, validando la condición adiabática y el gran tamaño del estanque. Sin embargo, la ola de frío penetra profundamente. Con el k eff de 26, la temperatura de 80°C cae (se desplaza a la izquierda) hasta 3 metros, lo que indica que casi la mitad del estanque está perdiendo calor activamente.

En ambos gráficos se observa el efecto de gradiente inverso cerca de la superficie. Esto representa la inercia térmica del fluido, el interior del estanque está desfasado temporalmente con la temperatura superficial.

La simulación demuestra que, incluso con un fluido de baja conductividad, los efectos combinados de la convección y un gradiente de temperatura extremo (de 80°C a -20°C) provocan pérdidas de calor significativas.

El análisis valida que el estanque, tal como está modelado (con una cara expuesta), no es adiabático en la práctica y pierde una enorme cantidad de energía. Esto justifica por qué, en el mundo real, los estanques de almacenamiento térmico deben estar completamente aislados en todas sus superficies.
