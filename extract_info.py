import numpy as np
import argparse

def process_map_closures(file_path):
    num_closures = 0
    inliers = []
    times = []
    
    with open(file_path, 'r') as file:
        for line in file:
            data = list(map(float, line.split()))
            if len(data) >= 22:  # Verificamos que la línea tiene suficientes valores
                num_closures += 1
                inliers.append(data[-2])  # Penúltimo valor
                times.append(data[-1])   # Último valor
    
    mean_inliers = np.mean(inliers) if inliers else 0
    mean_time = np.mean(times) if times else 0
    
    return num_closures, mean_inliers, mean_time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Procesar archivo map_closures.txt")
    parser.add_argument("file_path", type=str, help="Ruta del archivo map_closures.txt")
    args = parser.parse_args()
    
    num_closures, mean_inliers, mean_time = process_map_closures(args.file_path)
    
    print(f"Número de loop closures: {num_closures}")
    print(f"Número medio de inliers: {mean_inliers:.6f}")
    print(f"Tiempo medio de ejecución: {mean_time:.6f}")
