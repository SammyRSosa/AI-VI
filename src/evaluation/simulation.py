"""
evaluation/simulation.py — Simulador de Colas y Modelado Estocástico Monte Carlo.

Modela el comportamiento temporal del resolvedor híbrido Two-Phase bajo restricciones
de cuotas (RPM) del API, calculando intervalos de confianza estadísticos para
los esquemas Batch y Pairwise.
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple


class LLMQueueSimulator:
    """
    Simulador estocástico de eventos discretos para modelar la cola de peticiones al LLM
    y el comportamiento de la tasa de servicio (RPM) con penalizaciones por error 429.
    """

    def __init__(
        self,
        rpm_limit: int = 15,
        service_mu: float = 1.2,
        service_sigma: float = 0.2,
        backoff_seconds: float = 15.0,
    ):
        """
        Args:
            rpm_limit       : Capacidad de peticiones por minuto (RPM).
            service_mu      : Tiempo medio de servicio del LLM en segundos (distribución normal).
            service_sigma   : Desviación estándar del tiempo de servicio.
            backoff_seconds : Retardo / Penalización por error 429.
        """
        self.rpm_limit = rpm_limit
        self.service_mu = service_mu
        self.service_sigma = service_sigma
        self.backoff_seconds = backoff_seconds

    def simulate_run(self, num_requests: int) -> Dict[str, float]:
        """
        Simula una única corrida del sistema enviando `num_requests` de forma secuencial.
        Utiliza una ventana deslizante de 60 segundos para medir la tasa de arribo y
        detectar colisiones / bloqueos de tasa (error 429).
        
        Returns:
            Dict con métricas de tiempo total, bloqueos 429 y retardo por colas.
        """
        if num_requests <= 0:
            return {"total_time": 0.0, "blocks_triggered": 0.0, "queue_delay": 0.0}

        current_time = 0.0
        completed_timestamps: List[float] = []
        blocks_triggered = 0
        queue_delay = 0.0

        for _ in range(num_requests):
            # Limpiar timestamps de la ventana deslizante (> 60 segundos en el pasado)
            completed_timestamps = [t for t in completed_timestamps if current_time - t < 60.0]

            # Verificar si la ventana de 60 segundos está llena
            if len(completed_timestamps) >= self.rpm_limit:
                # Ocurre un bloqueo 429: el sistema debe dormir
                blocks_triggered += 1
                
                # Tiempo que falta para que expire el request más antiguo en la ventana
                oldest = completed_timestamps[0]
                wait_to_slide = 60.0 - (current_time - oldest)
                
                # Penalización: se espera lo que falta para limpiar el rate limit
                # o el tiempo de backoff fijo establecido por la API (el mayor)
                delay = max(self.backoff_seconds, wait_to_slide)
                
                queue_delay += delay
                current_time += delay
                
                # Volver a limpiar la ventana tras avanzar el tiempo
                completed_timestamps = [t for t in completed_timestamps if current_time - t < 60.0]

            # Generar el tiempo estocástico de servicio de la API (Normal truncada a >= 0.05s)
            service_time = max(0.05, random.normalvariate(self.service_mu, self.service_sigma))
            current_time += service_time
            completed_timestamps.append(current_time)

        return {
            "total_time": current_time,
            "blocks_triggered": float(blocks_triggered),
            "queue_delay": queue_delay,
        }

    def run_monte_carlo(
        self,
        num_requests: int,
        num_replicas: int = 50,
    ) -> Dict[str, Tuple[float, float, float]]:
        """
        Ejecuta múltiples réplicas Monte Carlo para obtener estimadores consistentes.
        Calcula el promedio y los intervalos de confianza del 95% usando la desviación estándar.
        
        Returns:
            Dict de métricas -> Tuple(promedio, límite_inferior_95, límite_superior_95)
        """
        times = []
        blocks = []
        delays = []

        for _ in range(num_replicas):
            res = self.simulate_run(num_requests)
            times.append(res["total_time"])
            blocks.append(res["blocks_triggered"])
            delays.append(res["queue_delay"])

        # Importar numpy localmente para evitar dependencias forzadas si se llama fuera de Streamlit
        import numpy as np

        def get_stats(data: List[float]) -> Tuple[float, float, float]:
            if not data:
                return 0.0, 0.0, 0.0
            mean = float(np.mean(data))
            std = float(np.std(data, ddof=1)) if len(data) > 1 else 0.0
            # Margen de error para el 95% de confianza (z=1.96)
            margin = 1.96 * (std / (len(data) ** 0.5)) if len(data) > 1 else 0.0
            return mean, max(0.0, mean - margin), mean + margin

        return {
            "total_time": get_stats(times),
            "blocks_triggered": get_stats(blocks),
            "queue_delay": get_stats(delays),
            "raw_times": (times, times, times)  # Retornar los datos crudos para histogramas
        }


def simulate_comparison(
    num_ambiguous_cuts: int,
    window_size: int,
    num_replicas: int = 50,
    rpm_limit: int = 15,
    service_mu: float = 1.2,
    service_sigma: float = 0.2,
    backoff_seconds: float = 15.0,
) -> Dict[str, dict]:
    """
    Función de utilidad para comparar directamente Batch y Pairwise.
    
    Cálculo del número de peticiones:
      - Batch: 1 petición por corte ambiguo.
      - Pairwise: (2 * window_size + 1) * 2 peticiones por corte ambiguo.
    """
    # Batch requests
    requests_batch = num_ambiguous_cuts
    
    # Pairwise requests: por cada corte se evalúan todos los puntos en el rango [cut-w, cut+w]
    # Cada punto candidato hace 2 evaluaciones de coherencia (izquierda y derecha)
    requests_pairwise = num_ambiguous_cuts * (2 * window_size + 1) * 2

    simulator = LLMQueueSimulator(
        rpm_limit=rpm_limit,
        service_mu=service_mu,
        service_sigma=service_sigma,
        backoff_seconds=backoff_seconds,
    )

    batch_results = simulator.run_monte_carlo(requests_batch, num_replicas=num_replicas)
    pairwise_results = simulator.run_monte_carlo(requests_pairwise, num_replicas=num_replicas)

    return {
        "params": {
            "num_ambiguous_cuts": num_ambiguous_cuts,
            "window_size": window_size,
            "requests_batch": requests_batch,
            "requests_pairwise": requests_pairwise,
            "num_replicas": num_replicas,
            "rpm_limit": rpm_limit,
            "service_mu": service_mu,
            "service_sigma": service_sigma,
        },
        "batch": batch_results,
        "pairwise": pairwise_results,
    }
