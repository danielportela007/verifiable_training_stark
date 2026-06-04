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

### Restricciones de Frontera Final (Boundary Constraints)
Valida que los pesos y sesgo finales del modelo entregado por el CSP coincidan con la última fila de la traza de ejecución auditada.

---

## Archivos del Código

* [toy_verifiable_training.py](file:///home/portela07/Projects/T%C3%A9sis/seccion2/verifiable_training_stark/toy_verifiable_training.py): Script de Python auto-contenido que implementa las clases `Client`, `CloudServiceProvider`, y `Verifier`, y ejecuta la demostración.
* [toy_execution_trace.json](file:///home/portela07/Projects/T%C3%A9sis/seccion2/verifiable_training_stark/toy_execution_trace.json): Estructura en formato JSON con la traza de ejecución detallada generada en el entrenamiento.
* [toy_execution_trace.csv](file:///home/portela07/Projects/T%C3%A9sis/seccion2/verifiable_training_stark/toy_execution_trace.csv): Representación tabular de la traza para auditorías matriciales rápidas.
