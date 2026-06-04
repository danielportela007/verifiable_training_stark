import csv
import hashlib
import json
import numpy as np
from typing import Dict, Any, Tuple, List


# =====================================================================
# PRIMITIVAS MATEMÁTICAS
# =====================================================================

def sigmoid(z: np.ndarray) -> np.ndarray:
    """Función de activación sigmoide (estable numéricamente)."""
    return np.where(
        z >= 0.0,
        1.0 / (1.0 + np.exp(-z)),
        np.exp(z) / (1.0 + np.exp(z))
    )


# =====================================================================
# ACTOR 1: CLIENTE (CLIENT)
# =====================================================================

class Client:
    """
    El Cliente es el dueño de los datos y del problema de negocio.
    Sus tareas son:
    1. Generar o recopilar el dataset.
    2. Generar un compromiso criptográfico del dataset (Merkle Root o Hash).
    3. Externalizar el entrenamiento al CSP.
    4. Solicitar la verificación de la prueba STARK del entrenamiento.
    """

    def __init__(self, random_state: int = 42):
        self.rng = np.random.default_rng(random_state)

    def generate_toy_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Genera un dataset de juguete muy pequeño:
        - 3 muestras (n = 3)
        - 2 características (d = 2)
        - Etiquetas binarias (y en {0, 1})
        """
        
        X = np.array([
            [0.5, -0.2],
            [-0.1, 0.8],
            [0.7, 0.3]
        ], dtype=float)
        
        y = np.array([1.0, 0.0, 1.0], dtype=float)
        
        print("\n[Cliente] Dataset generado:")
        for i in range(len(y)):
            print(f"  Muestra {i}: X = {X[i]}, y = {int(y[i])}")
            
        return X, y

    def commit_dataset(self, X: np.ndarray, y: np.ndarray) -> str:
        """
        Genera un compromiso hash SHA256 único del dataset.
        Esto asegura que el CSP entrene con EXACTAMENTE estos datos y no otros.
        """
        
        data_dict = {
            "X": X.tolist(),
            "y": y.tolist()
        }
        data_bytes = json.dumps(data_dict, sort_keys=True).encode('utf-8')
        dataset_hash = hashlib.sha256(data_bytes).hexdigest()
        print(f"[Cliente] Compromiso hash generado para el dataset: {dataset_hash[:16]}...")
        return dataset_hash


# =====================================================================
# ACTOR 2: CLOUD SERVICE PROVIDER (CSP)
# =====================================================================

class CloudServiceProvider:
    """
    El CSP es una entidad de cómputo poderosa pero no confiable (untrusted).
    Sus tareas son:
    1. Recibir el dataset (X, y) del Cliente.
    2. Entrenar el modelo de Regresión Logística (pesos w y bias b).
    3. Registrar cada paso de cómputo en una "Traza de Ejecución" (Execution Trace).
    4. Enviar los pesos entrenados, el bias y la traza de ejecución al Cliente/Verificador.
    """

    def __init__(self, learning_rate: float = 0.5, n_iterations: int = 4):
        self.lr = learning_rate
        self.n_iterations = n_iterations

    def train_and_generate_trace(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """
        Entrena el modelo usando Gradient Descent y registra detalladamente la
        traza de ejecución para emular el Prover de un sistema STARK.
        
        La traza es una matriz conceptual donde cada paso contiene:
        - Pesos (w) y bias (b) antes de la actualización.
        - Logits z y predicciones y_hat para cada muestra.
        - Gradientes calculados (grad_w, grad_b).
        - Pesos y bias nuevos (w_new, b_new).
        """
        n_samples, n_features = X.shape
        
        # Inicialización de parámetros (Restricción de frontera: w_0 = 0, b_0 = 0)
        w = np.zeros(n_features)
        b = 0.0
        
        # Estructura para registrar la traza de ejecución
        trace = {
            "steps": [],
            "metadata": {
                "n_samples": n_samples,
                "n_features": n_features,
                "learning_rate": self.lr,
                "n_iterations": self.n_iterations
            }
        }
        
        print(f"\n[CSP] Iniciando entrenamiento por {self.n_iterations} iteraciones...")
        
        for t in range(self.n_iterations):
            
            z = X @ w + b
            
            # 2. Activación (Sigmoide)
            y_hat = sigmoid(z)
            
            # 3. Pérdida (BCE)
            eps = 1e-15
            y_hat_c = np.clip(y_hat, eps, 1.0 - eps)
            loss = -np.mean(y * np.log(y_hat_c) + (1.0 - y) * np.log(1.0 - y_hat_c))
            
            # 4. Gradientes
            # dL/dw = (1/n) * X^T @ (y_hat - y)
            # dL/db = (1/n) * sum(y_hat - y)
            error = y_hat - y
            grad_w = (X.T @ error) / n_samples
            grad_b = np.sum(error) / n_samples
            
            # 5. Parámetros actualizados (Regla de transición)
            w_new = w - self.lr * grad_w
            b_new = b - self.lr * grad_b
            
            # Registrar este paso en la traza
            step_record = {
                "step": t,
                "w": w.copy().tolist(),
                "b": float(b),
                "z": z.tolist(),
                "y_hat": y_hat.tolist(),
                "loss": float(loss),
                "grad_w": grad_w.copy().tolist(),
                "grad_b": float(grad_b),
                "w_new": w_new.copy().tolist(),
                "b_new": float(b_new)
            }
            trace["steps"].append(step_record)
            
            print(f"  Paso {t}: Loss = {loss:.6f} | w = {w.round(4)} | b = {b:.4f} | grad_w = {grad_w.round(4)}")
            
            # Avanzar el estado
            w = w_new
            b = b_new
            
        # Añadimos el estado final de frontera en la traza
        trace["final_state"] = {
            "w": w.tolist(),
            "b": float(b)
        }
        
        print(f"[CSP] Entrenamiento completado. Modelo final: w = {w.round(4)}, b = {b:.4f}")
        return w, b, trace


# =====================================================================
# ACTOR 3: VERIFICADOR (VERIFIER)
# =====================================================================

class Verifier:
    """
    El Verificador es un agente (o smart contract) ligero y ultra rápido.
    Su tarea es verificar la prueba STARK de que el entrenamiento fue honesto.
    En lugar de re-entrenar (computar el modelo desde cero), el Verificador:
    1. Verifica la consistencia del Dataset con el compromiso hash (Boundary Constraint).
    2. Evalúa las Restricciones de Frontera (parámetros iniciales e inicialización).
    3. Evalúa las Restricciones de Transición (AIR) para cada fila 't' de la traza:
       - z[t] == X @ w[t] + b[t]
       - y_hat[t] == sigmoid(z[t])
       - grad_w[t] == X^T @ (y_hat[t] - y) / n
       - grad_b[t] == sum(y_hat[t] - y) / n
       - w[t+1] == w[t] - lr * grad_w[t]
       - b[t+1] == b[t] - lr * grad_b[t]
    """

    def __init__(self, tolerance: float = 1e-12):
        self.tol = tolerance

    def verify_dataset_commitment(self, X: np.ndarray, y: np.ndarray, expected_hash: str) -> bool:
        """Verifica que el dataset provisto coincida con el hash comprometido."""
        data_dict = {
            "X": X.tolist(),
            "y": y.tolist()
        }
        data_bytes = json.dumps(data_dict, sort_keys=True).encode('utf-8')
        computed_hash = hashlib.sha256(data_bytes).hexdigest()
        match = computed_hash == expected_hash
        print(f"[Verificador] Validación de compromiso de datos: {'ÉXITO' if match else 'FALLO'}")
        return match

    def verify_trace_air(self, X: np.ndarray, y: np.ndarray, trace: Dict[str, Any]) -> bool:
        """
        Evalúa las restricciones AIR (Algebraic Intermediate Representation)
        algebraicas sobre la traza de ejecución entregada por el CSP.
        """
        steps = trace["steps"]
        meta = trace["metadata"]
        n_samples = meta["n_samples"]
        n_features = meta["n_features"]
        lr = meta["learning_rate"]
        n_iterations = meta["n_iterations"]
        
        print("\n[Verificador] Iniciando auditoría aritmética de la traza (Restricciones AIR)...")
        
        # 1. Restricción de Frontera Inicial (Boundary Constraint at t=0)
        first_step = steps[0]
        init_w = np.array(first_step["w"])
        init_b = first_step["b"]
        
        if not np.allclose(init_w, 0.0, atol=self.tol) or abs(init_b) > self.tol:
            print(f"  [ERROR] Boundary Constraint falló en t=0: w_0={init_w}, b_0={init_b} (deben ser 0)")
            return False
        print("  [OK] Restricción de frontera inicial validada (w_0 = 0, b_0 = 0).")

        # 2. Restricciones de Transición (AIR Transition Constraints) para cada paso
        for t in range(n_iterations):
            step = steps[t]
            w = np.array(step["w"])
            b = step["b"]
            z = np.array(step["z"])
            y_hat = np.array(step["y_hat"])
            grad_w = np.array(step["grad_w"])
            grad_b = step["grad_b"]
            w_new = np.array(step["w_new"])
            b_new = step["b_new"]
            
            # A. Restricción AIR: Combinación Lineal
            # R_linear: z_i - (X_i . w + b) = 0
            computed_z = X @ w + b
            if not np.allclose(z, computed_z, atol=self.tol):
                print(f"  [ERROR] AIR roto en paso {t}: Logits (z) no coinciden con la combinación lineal.")
                return False
                
            # B. Restricción AIR: Activación
            # R_sigmoid: y_hat_i - sigmoid(z_i) = 0
            computed_yhat = sigmoid(z)
            if not np.allclose(y_hat, computed_yhat, atol=self.tol):
                print(f"  [ERROR] AIR roto en paso {t}: Predicciones (y_hat) no coinciden con la activación sigmoide.")
                return False
                
            # C. Restricción AIR: Gradientes
            # R_grad_w: grad_w - X^T @ (y_hat - y) / n = 0
            # R_grad_b: grad_b - sum(y_hat - y) / n = 0
            error = y_hat - y
            computed_grad_w = (X.T @ error) / n_samples
            computed_grad_b = np.sum(error) / n_samples
            
            if not np.allclose(grad_w, computed_grad_w, atol=self.tol):
                print(f"  [ERROR] AIR roto en paso {t}: grad_w incorrecto.")
                return False
            if abs(grad_b - computed_grad_b) > self.tol:
                print(f"  [ERROR] AIR roto en paso {t}: grad_b incorrecto.")
                return False

            # D. Restricción AIR: Actualización de Estado (Paso de tiempo t -> t+1)
            # R_transition_w: w_new - (w - lr * grad_w) = 0
            # R_transition_b: b_new - (b - lr * grad_b) = 0
            computed_wnew = w - lr * grad_w
            computed_bnew = b - lr * grad_b
            if not np.allclose(w_new, computed_wnew, atol=self.tol):
                print(f"  [ERROR] AIR de transición roto en paso {t}: w_new incorrecto.")
                return False
            if abs(b_new - computed_bnew) > self.tol:
                print(f"  [ERROR] AIR de transición roto en paso {t}: b_new incorrecto.")
                return False
                
            # E. Restricción AIR: Continuidad de Memoria (t -> t+1 en filas contiguas)
            # Si no es el último paso, el w del paso t+1 debe ser igual al w_new del paso t
            if t < n_iterations - 1:
                next_step = steps[t + 1]
                next_w = np.array(next_step["w"])
                next_b = next_step["b"]
                if not np.allclose(next_w, w_new, atol=self.tol) or abs(next_b - b_new) > self.tol:
                    print(f"  [ERROR] Restricción de continuidad rota entre paso {t} y {t+1}.")
                    return False

        print("  [OK] Todas las restricciones de transición AIR se cumplen perfectamente.")

        # 3. Restricción de Frontera Final (Boundary Constraint at t=T)
        final_state = trace["final_state"]
        final_w = np.array(final_state["w"])
        final_b = final_state["b"]
        
        last_step = steps[-1]
        last_wnew = np.array(last_step["w_new"])
        last_bnew = last_step["b_new"]
        
        if not np.allclose(final_w, last_wnew, atol=self.tol) or abs(final_b - last_bnew) > self.tol:
            print("  [ERROR] Boundary Constraint falló para el estado final.")
            return False
            
        print("  [OK] Restricción de frontera final validada con éxito.")
        print("[Verificador] ¡Auditoría completada! La traza es 100% VÁLIDA y matemáticamente íntegra.")
        return True


# =====================================================================
# EJECUCIÓN PRINCIPAL: DEMOSTRACIÓN END-TO-END
# =====================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DEMOSTRACIÓN DE APRENDIZAJE AUTOMÁTICO VERIFICABLE (zkML / STARKs)")
    print("CASO DE ESTUDIO: REGRESIÓN LOGÍSTICA (EJEMPLO DE JUGUETE)")
    print("=" * 70)
    
    # 1. El Cliente prepara el escenario
    client = Client()
    X_toy, y_toy = client.generate_toy_dataset()
    commitment = client.commit_dataset(X_toy, y_toy)
    
    # 2. El CSP ejecuta el cómputo y genera la traza
    # Usamos learning rate alto (0.5) y 4 iteraciones para ver cambios claros
    csp = CloudServiceProvider(learning_rate=0.5, n_iterations=4)
    w_opt, b_opt, trace_output = csp.train_and_generate_trace(X_toy, y_toy)
    
    
    trace_filepath = "toy_execution_trace.json"
    with open(trace_filepath, "w") as f:
        json.dump(trace_output, f, indent=2)
    print(f"[Sistema] Traza de ejecución serializada en: {trace_filepath}")

    
    csv_filepath = "toy_execution_trace.csv"
    n_features = trace_output["metadata"]["n_features"]
    n_samples   = trace_output["metadata"]["n_samples"]

    
    fieldnames = (
        ["step"]
        + [f"w_{j}" for j in range(n_features)]
        + ["b"]
        + [f"z_{i}" for i in range(n_samples)]
        + [f"y_hat_{i}" for i in range(n_samples)]
        + ["loss"]
        + [f"grad_w_{j}" for j in range(n_features)]
        + ["grad_b"]
        + [f"w_new_{j}" for j in range(n_features)]
        + ["b_new"]
    )

    with open(csv_filepath, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for s in trace_output["steps"]:
            row = {"step": s["step"]}
            for j in range(n_features):
                row[f"w_{j}"]     = s["w"][j]
                row[f"w_new_{j}"] = s["w_new"][j]
                row[f"grad_w_{j}"] = s["grad_w"][j]
            row["b"]      = s["b"]
            row["b_new"]  = s["b_new"]
            row["grad_b"] = s["grad_b"]
            row["loss"]   = s["loss"]
            for i in range(n_samples):
                row[f"z_{i}"]     = s["z"][i]
                row[f"y_hat_{i}"] = s["y_hat"][i]
            writer.writerow(row)

    print(f"[Sistema] Traza exportada como matriz CSV en: {csv_filepath}")
    
    # 3. El Verificador recibe el dataset, la traza y el compromiso para auditoría
    verifier = Verifier()
    
    # A. Caso Correcto: El Verificador evalúa al CSP honesto
    print("\n--- CASO 1: VERIFICANDO COMPUTACIÓN HONESTA ---")
    dataset_ok = verifier.verify_dataset_commitment(X_toy, y_toy, commitment)
    if dataset_ok:
        trace_ok = verifier.verify_trace_air(X_toy, y_toy, trace_output)
        if trace_ok:
            print("\n>>> RESULTADO: El Cliente ACEPTA el modelo entrenado con total seguridad.")
            print(f"    Pesos finales: {w_opt.round(6)}, Sesgo final: {b_opt:.6f}")
        else:
            print("\n>>> RESULTADO: El Cliente RECHAZA el modelo por traza inválida.")
    else:
        print("\n>>> RESULTADO: El Cliente RECHAZA los datos presentados por alteración de hash.")

    # B. Caso Malicioso: ¿Qué pasa si el CSP intenta engañar al Cliente enviando
    # pesos optimizados falsos o alterando la traza?
    print("\n--- CASO 2: INTENTO DE ATAQUE DEL CSP (TRAZA ALTERADA) ---")
    # El CSP malicioso altera discretamente un peso en el paso 2 para ahorrar cómputo
    malicious_trace = json.loads(json.dumps(trace_output)) # deep copy
    malicious_trace["steps"][2]["w_new"][0] += 0.05  # Alteración maliciosa (inyección de backdoor o flojera)
    
    dataset_ok = verifier.verify_dataset_commitment(X_toy, y_toy, commitment)
    if dataset_ok:
        print("[Sistema] Enviando traza alterada al verificador...")
        trace_ok = verifier.verify_trace_air(X_toy, y_toy, malicious_trace)
        if trace_ok:
            print("\n>>> RESULTADO: [CRÍTICO] El Verificador fue engañado (Falso Positivo).")
        else:
            print("\n>>> RESULTADO: El Cliente RECHAZA exitosamente el modelo malicioso. ¡El engaño fue detectado!")
    
    print("\n" + "=" * 70)
    print("FIN DE LA DEMOSTRACIÓN")
    print("=" * 70)
