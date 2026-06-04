# Documentación Técnica: Entrenamiento Verificable de Regresión Logística (zkML / STARKs)

Este repositorio contiene la implementación y simulación en Python de un sistema de **Entrenamiento Verificable (Verifiable Training)** para Regresión Logística, inspirado en el funcionamiento de los sistemas **zk-STARK** (Zero-Knowledge Succinct Non-Interactive Argument of Knowledge) aplicados a Aprendizaje Automático (zkML).

El objetivo es permitir a un **Cliente** delegar el cómputo del entrenamiento de un modelo a un proveedor en la nube (**CSP**) potencialmente malicioso o no confiable (*untrusted*), y verificar de manera ultra rápida y en tiempo constante si el CSP ejecutó el entrenamiento de manera honesta y correcta sobre los datos originales.

---

## 🏛️ Arquitectura del Sistema

La arquitectura está compuesta por tres actores principales con responsabilidades específicas:

```
                  ┌─────────────────────────────────────────────────────┐
                  │                   CLIENTE                           │
                  │  DataPrep ──► DataCommit ──► ClientControl          │
                  │       │              │                 ▲            │
                  │       │ (X, y)       │ hash_c          │ veredicto  │
                  └───────┼──────────────┼─────────────────┼────────────┘
                          │              │                 │
                          ▼              ▼                 │
                  ┌───────────────┐      │      ┌──────────┴───────────┐
                  │     CSP       │      │      │     VERIFICADOR      │
                  │ TrainingEngine│      │      │  CommitValidator     │
                  │ TraceGenerator│      └─────►│  AIREngine           │
                  │ ProverMock    │─────────────►│                     │
                  └───────────────┘  traza JSON  └─────────────────────┘
```

### 1. Cliente (Client)
Es el propietario de los datos de entrenamiento ($\mathbf{X}$, $\mathbf{y}$). Sus tareas son:
- **`DataPrep`**: Preparar las características y etiquetas.
- **`DataCommit`**: Generar un compromiso criptográfico único (Hash SHA-256) del dataset original para evitar que el CSP entrene con datos manipulados.
- **`ClientControl`**: Coordinar y solicitar la auditoría.

### 2. Cloud Service Provider (CSP)
Es el proveedor de cómputo en la nube (emula al **Prover** en un sistema STARK). Sus tareas son:
- Recibir el dataset original del Cliente.
- **`TrainingEngine`**: Ejecutar el entrenamiento usando Descenso de Gradiente (Gradient Descent).
- **`TraceGenerator`**: Registrar detalladamente todas las variables intermedias de cada iteración en una matriz bidimensional llamada **Traza de Ejecución** (*Execution Trace*).
- Exportar los parámetros óptimos finales y la traza para auditoría en formatos JSON y CSV.

### 3. Verificador (Verifier)
Es un agente ligero (emula al **Verifier** de un sistema STARK) que valida la prueba en tiempo constante sin re-entrenar el modelo. Sus tareas son:
- **`CommitValidator`**: Validar que el dataset auditado coincida con el compromiso hash enviado por el Cliente.
- **`AIREngine`**: Evaluar un conjunto de **Restricciones Intermedias Algebraicas (AIR)** sobre la traza de ejecución provista.

---

## 📐 Restricciones Aritméticas (AIR - Algebraic Intermediate Representation)

El Verificador comprueba la corrección matemática de la traza de ejecución evaluando tres tipos de restricciones:

### A. Restricciones de Frontera Inicial (Boundary Constraints)
Asegura que el modelo inició en un estado predefinido y no sesgado:
$$\mathbf{w}_0 = \mathbf{0}, \quad b_0 = 0$$

### B. Restricciones de Transición (Transition Constraints)
Para cada iteración $t \in [0, T-1]$, el Verificador evalúa:
1. **Combinación Lineal**: $\mathbf{z}^{(t)} = \mathbf{X}\mathbf{w}^{(t)} + b^{(t)}$
2. **Activación**: $\hat{\mathbf{y}}^{(t)} = \sigma(\mathbf{z}^{(t)})$ donde $\sigma(z) = \frac{1}{1 + e^{-z}}$
3. **Cálculo de Gradientes**:
   $$\mathbf{g}_w^{(t)} = \frac{1}{n} \mathbf{X}^T (\hat{\mathbf{y}}^{(t)} - \mathbf{y})$$
   $$g_b^{(t)} = \frac{1}{n} \sum_{i=1}^n (\hat{y}_i^{(t)} - y_i)$$
4. **Regla de Transición**: Actualización de los pesos usando la tasa de aprendizaje $\eta$:
   $$\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \eta \, \mathbf{g}_w^{(t)}$$
   $$b^{(t+1)} = b^{(t)} - \eta \, g_b^{(t)}$$
5. **Continuidad de Memoria**: El estado inicial del paso $t+1$ debe coincidir exactamente con el estado final calculado en el paso $t$.

### C. Restricciones de Frontera Final (Boundary Constraints)
Valida que los pesos y sesgo finales del modelo entregado por el CSP coincidan con la última fila de la traza de ejecución auditada.

---

## 📂 Archivos del Código

* [toy_verifiable_training.py](file:///home/portela07/Projects/T%C3%A9sis/seccion2/verifiable_training_stark/toy_verifiable_training.py): Script de Python auto-contenido que implementa las clases `Client`, `CloudServiceProvider`, y `Verifier`, y ejecuta la demostración.
* [toy_execution_trace.json](file:///home/portela07/Projects/T%C3%A9sis/seccion2/verifiable_training_stark/toy_execution_trace.json): Estructura en formato JSON con la traza de ejecución detallada generada en el entrenamiento.
* [toy_execution_trace.csv](file:///home/portela07/Projects/T%C3%A9sis/seccion2/verifiable_training_stark/toy_execution_trace.csv): Representación tabular de la traza para auditorías matriciales rápidas.

---

## 🚀 Instrucciones de Ejecución

Para correr la demostración interactiva que simula tanto un entrenamiento honesto como una detección de fraude en la traza:

1. Asegúrate de tener instalado `numpy`:
   ```bash
   pip install numpy
   ```
2. Ejecuta el archivo principal:
   ```bash
   python3 toy_verifiable_training.py
   ```

El programa imprimirá en la terminal el progreso de los actores y los resultados de ambas auditorías (éxito en entrenamiento honesto y detección de anomalía en traza maliciosa).
